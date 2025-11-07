"""Pretraining pipeline for TCN-based VAE + detector on windowed time-series data."""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    load_adbench_data,
    train_detector,
    evaluate_with_classification_report_and_auc,
    total_anomaly_vae_loss,
)
from tcn_models import TCNVAE, WindowDataProcessor, TransformerDetector


def load_dataset(dataset_path: str) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Load sequential dataset and return splits plus optional segment boundaries."""
    data = np.load(dataset_path, allow_pickle=True)
    has_pre_split = {"X_train", "y_train", "X_test", "y_test"} <= set(data.files)

    if has_pre_split:
        X_train_np = data["X_train"]
        y_train_np = data["y_train"]
        X_test_np = data["X_test"]
        y_test_np = data["y_test"]
    else:
        X_all, y_all = load_adbench_data(dataset_path)
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_all.numpy(),
            y_all.numpy(),
            test_size=0.6,
            random_state=42,
            stratify=y_all,
        )

    train_segments = data["train_segments"] if "train_segments" in data.files else None
    test_segments = data["test_segments"] if "test_segments" in data.files else None

    return X_train_np, y_train_np, X_test_np, y_test_np, train_segments, test_segments


def ensure_2d(array: np.ndarray) -> np.ndarray:
    """Ensure the array has shape (N, C)."""
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def create_windows_from_segments(
    values: np.ndarray,
    labels: Optional[np.ndarray],
    segments: Optional[np.ndarray],
    window_size: int,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Generate windows without crossing segment boundaries."""
    values = ensure_2d(values)

    if labels is None:
        labels_array = np.zeros(values.shape[0], dtype=np.int8)
    else:
        labels_array = labels.reshape(-1)
        if labels_array.shape[0] != values.shape[0]:
            raise ValueError("Labels and values must share the same length")

    if segments is None or len(segments) == 0:
        segments_iter = [(0, values.shape[0])]
    else:
        segments_iter = [(int(start), int(length)) for start, length in segments]

    windows: List[np.ndarray] = []
    window_labels: List[float] = []
    skipped_segments = 0

    for start, length in segments_iter:
        end = start + length
        segment = values[start:end]
        segment_labels = labels_array[start:end]

        if segment.shape[0] < window_size:
            skipped_segments += 1
            continue

        for offset in range(0, segment.shape[0] - window_size + 1, stride):
            window = segment[offset : offset + window_size]
            windows.append(window.T)  # (features, window_size)
            window_label = float(np.max(segment_labels[offset : offset + window_size]))
            window_labels.append(window_label)

    if not windows:
        # Return empty tensors to keep downstream code resilient
        empty_windows = torch.zeros((0, values.shape[1], window_size), dtype=torch.float32)
        empty_labels = torch.zeros((0, 1), dtype=torch.float32)
        return empty_windows, empty_labels, skipped_segments

    windows_tensor = torch.tensor(np.stack(windows), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(window_labels).reshape(-1, 1), dtype=torch.float32)
    return windows_tensor, labels_tensor, skipped_segments


def tcn_vae_loss(x, x_recon, mu, logvar, beta):
    """Reconstruction + KL divergence loss for TCN-VAE."""
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain TCN-VAE + detector on sequential dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ADBench_datasets/UCR_Anomaly_FullData_pretrain.npz",
        help="Path to sequential dataset (.npz) containing X_train/X_test splits",
    )
    parser.add_argument("--window-size", type=int, default=128, help="Sliding window length")
    parser.add_argument("--window-stride", type=int, default=32, help="Sliding window stride")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for loaders")
    parser.add_argument("--vae-epochs", type=int, default=450, help="Number of TCN-VAE epochs")
    parser.add_argument("--detector-epochs", type=int, default=50, help="Number of detector epochs")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of VAE windows reserved for validation (0 disables validation split)",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for perturbation loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for zero-perturbation loss")
    parser.add_argument("--zeta", type=float, default=1.0, help="Weight for enhanced KL loss")
    parser.add_argument("--delta-min", type=float, default=0.1, help="Triplet margin lower bound")
    parser.add_argument("--delta-max", type=float, default=1.0, help="Triplet margin upper bound")
    parser.add_argument("--sigma-prior", type=float, default=0.5, help="Prior std for enhanced KL")
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=50,
        help="Number of epochs to anneal KL weight from zeta_start to zeta",
    )
    parser.add_argument(
        "--zeta-start",
        type=float,
        default=0.05,
        help="Initial KL weight before warmup reaches full zeta",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Epochs to wait for improvement before early stopping (0 disables)",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum loss improvement to reset patience",
    )
    return parser.parse_args()


def main(
    dataset_path: str,
    window_size: int = 128,
    window_stride: int = 32,
    batch_size: int = 64,
    num_epochs_vae: int = 450,
    num_epochs_detector: int = 50,
    alpha_recon: float = 1.0,
    beta_perturb: float = 1.0,
    gamma_zero: float = 1.0,
    zeta_en_kl: float = 0.9,
    kl_warmup_epochs: int = 50,
    zeta_start: float = 0.05,
    delta_min: float = 0.1,
    delta_max: float = 1.0,
    sigma_prior: float = 0.5,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    val_fraction: float = 0.1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load and standardize sequential data
    dataset_path = str(dataset_path)
    (
        X_train_np,
        y_train_np,
        X_test_np,
        y_test_np,
        train_segments,
        test_segments,
    ) = load_dataset(dataset_path)
    X_train_np = ensure_2d(X_train_np)
    X_test_np = ensure_2d(X_test_np)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)

    y_train_np = y_train_np.reshape(-1)
    y_test_np = y_test_np.reshape(-1)

    # 2. Window the time series
    window_processor = WindowDataProcessor(window_size=window_size, stride=window_stride)

    if train_segments is not None and len(train_segments) > 0:
        train_windows, train_window_labels, skipped_train = create_windows_from_segments(
            X_train_scaled,
            y_train_np,
            train_segments,
            window_size,
            window_stride,
        )
        if skipped_train:
            print(f"Skipped {skipped_train} train segments shorter than window_size={window_size}")
    else:
        train_series = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(0)
        train_labels_seq = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(0)
        train_windows, train_window_labels = window_processor.create_windows(train_series, train_labels_seq)
        skipped_train = 0

    if test_segments is not None and len(test_segments) > 0:
        test_windows, test_window_labels, skipped_test = create_windows_from_segments(
            X_test_scaled,
            y_test_np,
            test_segments,
            window_size,
            window_stride,
        )
        if skipped_test:
            print(f"Skipped {skipped_test} test segments shorter than window_size={window_size}")
    else:
        test_series = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(0)
        test_labels_seq = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(0)
        test_windows, test_window_labels = window_processor.create_windows(test_series, test_labels_seq)
        skipped_test = 0

    if train_windows.numel() == 0:
        raise RuntimeError(
            "No training windows were generated. Consider reducing window_size or verify dataset preparation."
        )

    if test_windows.numel() == 0:
        raise RuntimeError(
            "No test windows were generated. Consider reducing window_size or verify dataset preparation."
        )

    input_channels = train_windows.shape[1]

    # 3. Train TCN-VAE on windowed data
    tcn_vae = TCNVAE(
        input_channels=input_channels,
        window_size=window_size,
        latent_dim=64,
        tcn_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.1,
        beta=1.0,
        use_multiscale=True,
        multiscale_kernels=[3, 5, 7],
    ).to(device)

    vae_dataset = TensorDataset(train_windows, train_window_labels)
    val_loader = None
    train_dataset_for_vae = vae_dataset

    if 0.0 < val_fraction < 1.0 and len(vae_dataset) > 1:
        proposed_val = max(1, int(len(vae_dataset) * val_fraction))
        if proposed_val >= len(vae_dataset):
            proposed_val = len(vae_dataset) - 1

        train_size = len(vae_dataset) - proposed_val
        if train_size <= 0:
            train_size = len(vae_dataset)
            proposed_val = 0

        if proposed_val > 0:
            generator = torch.Generator().manual_seed(42)
            train_dataset_for_vae, val_dataset = random_split(
                vae_dataset,
                [train_size, proposed_val],
                generator=generator,
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vae_loader = DataLoader(train_dataset_for_vae, batch_size=batch_size, shuffle=True)

    optimizer_vae = Adam(tcn_vae.parameters(), lr=5e-5)

    best_metric = float("inf")
    epochs_since_improve = 0

    for epoch in range(num_epochs_vae):
        if kl_warmup_epochs <= 0:
            kl_weight = zeta_en_kl
        else:
            warm_ratio = min(1.0, (epoch + 1) / kl_warmup_epochs)
            kl_weight = zeta_start + (zeta_en_kl - zeta_start) * warm_ratio

        tcn_vae.train()
        total_loss = 0.0
        total_recon = 0.0
        total_perturb = 0.0
        total_zero = 0.0
        total_kl = 0.0
        for window_batch, label_batch in vae_loader:
            window_batch = window_batch.to(device)
            label_batch = label_batch.to(device)

            recon, mu, logvar = tcn_vae(window_batch, label_batch)
            loss, recon_l, perturb_l, zero_l, kl_l = total_anomaly_vae_loss(
                x=window_batch,
                x_recon=recon,
                mu=mu,
                logvar=logvar,
                x_tilde=None,
                alpha=alpha_recon,
                beta=beta_perturb,
                gamma=gamma_zero,
                zeta=kl_weight,
                delta_min=delta_min,
                delta_max=delta_max,
                sigma_prior=sigma_prior,
            )

            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()

            total_loss += float(loss.item())
            total_recon += float(recon_l.item())
            total_perturb += float(perturb_l.item())
            total_zero += float(zero_l.item())
            total_kl += float(kl_l.item())

        denominator = max(len(vae_loader), 1)
        avg_loss = total_loss / denominator
        avg_recon = total_recon / denominator
        avg_perturb = total_perturb / denominator
        avg_zero = total_zero / denominator
        avg_kl = total_kl / denominator

        val_metrics = None
        if val_loader is not None:
            tcn_vae.eval()
            val_total = 0.0
            val_recon = 0.0
            val_perturb = 0.0
            val_zero = 0.0
            val_kl = 0.0
            with torch.no_grad():
                for window_batch, label_batch in val_loader:
                    window_batch = window_batch.to(device)
                    label_batch = label_batch.to(device)

                    recon, mu, logvar = tcn_vae(window_batch, label_batch)
                    loss_v, recon_v, perturb_v, zero_v, kl_v = total_anomaly_vae_loss(
                        x=window_batch,
                        x_recon=recon,
                        mu=mu,
                        logvar=logvar,
                        x_tilde=None,
                        alpha=alpha_recon,
                        beta=beta_perturb,
                        gamma=gamma_zero,
                        zeta=kl_weight,
                        delta_min=delta_min,
                        delta_max=delta_max,
                        sigma_prior=sigma_prior,
                    )

                    val_total += float(loss_v.item())
                    val_recon += float(recon_v.item())
                    val_perturb += float(perturb_v.item())
                    val_zero += float(zero_v.item())
                    val_kl += float(kl_v.item())

            denom_val = max(len(val_loader), 1)
            val_metrics = {
                "loss": val_total / denom_val,
                "recon": val_recon / denom_val,
                "perturb": val_perturb / denom_val,
                "zero": val_zero / denom_val,
                "kl": val_kl / denom_val,
            }

        if (epoch + 1) % 10 == 0:
            message = (
                f"[TCN-VAE] Epoch {epoch + 1}/{num_epochs_vae}, "
                f"train_total={avg_loss:.4f}, recon={avg_recon:.4f}, "
                f"perturb={avg_perturb:.4f}, zero={avg_zero:.4f}, kl={avg_kl:.4f}, "
                f"kl_weight={kl_weight:.4f}"
            )
            if val_metrics is not None:
                message += (
                    f" | val_total={val_metrics['loss']:.4f}, val_recon={val_metrics['recon']:.4f}, "
                    f"val_perturb={val_metrics['perturb']:.4f}, val_zero={val_metrics['zero']:.4f}, "
                    f"val_kl={val_metrics['kl']:.4f}"
                )
            print(message)

        metric_loss = val_metrics["loss"] if val_metrics is not None else avg_loss

        if metric_loss < best_metric - early_stop_min_delta:
            best_metric = metric_loss
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if early_stop_patience > 0 and epochs_since_improve >= early_stop_patience:
            print(
                f"[TCN-VAE] Early stopping at epoch {epoch + 1} after no improvement for {early_stop_patience} epochs\n"
                f"            Best monitored loss: {best_metric:.4f}"
            )
            break

    # 4. Generate synthetic minority windows
    with torch.no_grad():
        labels_flat = train_window_labels.squeeze(1)
        minority_mask = labels_flat == 1
        majority_mask = labels_flat == 0

        minority_windows = train_windows[minority_mask]
        majority_windows = train_windows[majority_mask]

        num_generate = max(0, len(majority_windows) - len(minority_windows))

        if num_generate > 0 and len(minority_windows) > 0:
            z_samples = (torch.rand(num_generate, tcn_vae.latent_dim) * 4.0) - 2.0
            z_samples = z_samples.to(device)
            y_synthetic = torch.full((num_generate, 1), 0.9, device=device)
            synthetic_windows = tcn_vae.decode(z_samples, y_synthetic).cpu()
            synthetic_labels = torch.ones(num_generate, 1)

            augmented_windows = torch.cat([train_windows, synthetic_windows], dim=0)
            augmented_labels = torch.cat([train_window_labels, synthetic_labels], dim=0)
        else:
            augmented_windows = train_windows
            augmented_labels = train_window_labels

    # 5. Prepare data for detector (flatten windows)
    train_features = augmented_windows.reshape(augmented_windows.size(0), -1)
    train_labels = augmented_labels.squeeze(1)

    test_features = test_windows.reshape(test_windows.size(0), -1)
    test_labels = test_window_labels.squeeze(1)

    detector_input_size = train_features.size(1)

    train_dataset_final = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    train_loader_final = DataLoader(train_dataset_final, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    print("After oversampling using TCN-VAE:")
    unique, counts = np.unique(train_labels.numpy(), return_counts=True)
    print("Class distribution in the windowed training set:", dict(zip(unique, counts)))

    detector = TransformerDetector(input_size=detector_input_size).to(device)
    optimizer_detector = Adam(detector.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs_detector):
        train_loss = train_detector(detector, train_loader_final, optimizer_detector, criterion, device)
        if (epoch + 1) % 5 == 0:
            print(f"[Detector] Epoch {epoch + 1}/{num_epochs_detector}, Loss={train_loss:.4f}")

    evaluate_with_classification_report_and_auc(detector, test_loader, device, threshold=0.5)

    # 6. Persist models and preprocessing metadata
    dataset_name = Path(dataset_path).stem
    save_dir = Path("./saved_models") / dataset_name
    os.makedirs(save_dir, exist_ok=True)

    vae_path = save_dir / "tcn_vae.pth"
    detector_path = save_dir / "tcn_transformer_detector.pth"
    meta_path = save_dir / "tcn_training_meta.pt"

    torch.save(tcn_vae.state_dict(), vae_path)
    torch.save(detector.state_dict(), detector_path)

    meta = {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "window_size": window_size,
        "window_stride": window_stride,
        "input_channels": input_channels,
        "detector_input_size": detector_input_size,
    }
    torch.save(meta, meta_path)

    print(f"[{dataset_name}] TCN-VAE model saved to: {vae_path}")
    print(f"[{dataset_name}] Detector model saved to: {detector_path}")
    print(f"[{dataset_name}] Training metadata saved to: {meta_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        num_epochs_vae=args.vae_epochs,
        num_epochs_detector=args.detector_epochs,
        alpha_recon=args.alpha,
        beta_perturb=args.beta,
        gamma_zero=args.gamma,
        zeta_en_kl=args.zeta,
        kl_warmup_epochs=args.kl_warmup_epochs,
        zeta_start=args.zeta_start,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        sigma_prior=args.sigma_prior,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        val_fraction=args.val_fraction,
    )
