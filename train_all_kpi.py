"""Run Swift Hydra TCN pipeline across all KPI datasets in KPI_npz directory."""

import argparse
import traceback
from pathlib import Path

from pretrain_tcn import main as pretrain_main
from SwiftHydra_tcn import main as hydra_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swift Hydra TCN pipeline across KPI datasets")
    parser.add_argument("--dataset-dir", type=str, default="KPI_npz", help="Directory with KPI .npz files")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vae-epochs", type=int, default=450)
    parser.add_argument("--detector-epochs", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--gen-windows", type=int, default=32)
    parser.add_argument("--episode-vae-epochs", type=int, default=5)
    parser.add_argument("--episode-detector-epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--delta-min", type=float, default=0.1)
    parser.add_argument("--delta-max", type=float, default=1.0)
    parser.add_argument("--sigma-prior", type=float, default=0.5)
    parser.add_argument("--loss-weight", type=float, default=0.1)
    parser.add_argument("--adv-steps", type=int, default=25)
    parser.add_argument("--adv-lr", type=float, default=0.01)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pretraining stage")
    parser.add_argument("--skip-adaptive", action="store_true", help="Skip adaptive Swift Hydra stage")
    return parser.parse_args()


def collect_datasets(dataset_dir: Path) -> list[str]:
    datasets = []
    for path in sorted(dataset_dir.glob("*.npz")):
        if path.name.startswith("_index"):
            continue
        datasets.append(str(path))
    return datasets


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    datasets = collect_datasets(dataset_dir)
    if not datasets:
        raise RuntimeError(f"No .npz datasets found in {dataset_dir}")

    print(f"Discovered {len(datasets)} KPI datasets in {dataset_dir}.")

    successes = []
    failures = []

    for dataset_path in datasets:
        dataset_name = Path(dataset_path).stem
        print("=" * 80)
        print(f"Starting pipeline for {dataset_name}")
        try:
            if not args.skip_pretrain:
                pretrain_main(
                    dataset_path=dataset_path,
                    window_size=args.window_size,
                    window_stride=args.window_stride,
                    batch_size=args.batch_size,
                    num_epochs_vae=args.vae_epochs,
                    num_epochs_detector=args.detector_epochs,
                    alpha_recon=args.alpha,
                    beta_perturb=args.beta,
                    gamma_zero=args.gamma,
                    zeta_en_kl=args.zeta,
                    delta_min=args.delta_min,
                    delta_max=args.delta_max,
                    sigma_prior=args.sigma_prior,
                    early_stop_patience=args.early_stop_patience,
                    early_stop_min_delta=args.early_stop_min_delta,
                    val_fraction=args.val_fraction,
                )

            if not args.skip_adaptive:
                hydra_main(
                    dataset_path=dataset_path,
                    num_episodes=args.episodes,
                    num_gen_windows=args.gen_windows,
                    batch_size=args.batch_size,
                    vae_epochs_per_episode=args.episode_vae_epochs,
                    detector_epochs_per_episode=args.episode_detector_epochs,
                    num_epochs_final=args.final_epochs,
                    alpha_recon=args.alpha,
                    beta_perturb=args.beta,
                    gamma_zero=args.gamma,
                    zeta_en_kl=args.zeta,
                    delta_min=args.delta_min,
                    delta_max=args.delta_max,
                    sigma_prior=args.sigma_prior,
                    total_loss_weight=args.loss_weight,
                    adv_steps=args.adv_steps,
                    adv_lr=args.adv_lr,
                )

            successes.append(dataset_name)
        except Exception as exc:  # noqa: BLE001
            print(f"Pipeline failed for {dataset_name}: {exc}")
            traceback.print_exc()
            failures.append(dataset_name)

    print("=" * 80)
    print("Pipeline summary:")
    print(f"  Successes ({len(successes)}): {', '.join(successes) if successes else 'None'}")
    print(f"  Failures ({len(failures)}): {', '.join(failures) if failures else 'None'}")


if __name__ == "__main__":
    main()
