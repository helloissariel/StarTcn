"""Adaptive training loop for TCN-VAE based Swift Hydra pipeline."""

import argparse
import copy
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    load_adbench_data,
    train_detector,
    evaluate_with_classification_report_and_auc,
    total_anomaly_vae_loss,
)
from tcn_models import (
    TCNVAE,
    WindowDataProcessor,
    TransformerDetector,
)
from model import PolicyNetwork, ValueNetwork, PPOTrainer

try:
    import gym
except ImportError:  # pragma: no cover
    import gymnasium as gym  # type: ignore

from gym import spaces

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import RecurrentPPO
    SB3_AVAILABLE = True
except ImportError:  # pragma: no cover
    SB3_AVAILABLE = False

from merlion.evaluate.anomaly import ScoreType
from merlion_evaluator import MerlionEvaluator


def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(dataset_path)
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

    return X_train_np, y_train_np, X_test_np, y_test_np


def ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def build_scaler_from_meta(meta: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(meta["scaler_mean"])
    scaler.scale_ = np.asarray(meta["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    scaler.n_samples_seen_ = np.array([1], dtype=np.float64)
    return scaler


def evaluate_detector_merlion(
    detector: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    evaluator: MerlionEvaluator,
) -> tuple[float, float, dict]:
    """Compute RPA/PA F1 scores via Merlion evaluator."""
    detector.eval()
    score_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = detector(X_batch).view(-1)
            score_batches.append(preds.detach().cpu())
            label_batches.append(y_batch.detach().cpu().view(-1))

    if not score_batches:
        return 0.0, 0.0, {}

    scores = torch.cat(score_batches).numpy()
    labels = torch.cat(label_batches).numpy()

    merlion_result = evaluator.evaluate_comprehensive(scores, labels, optimization_method="floating")
    rpa_f1 = float(merlion_result.get("rpa_f1", 0.0))
    pa_f1 = float(merlion_result.get("pa_f1", 0.0))
    return rpa_f1, pa_f1, merlion_result


def log_best_metrics(dataset_name: str, metrics: dict) -> None:
    """Append best metrics for a dataset to a CSV log."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "tcn_best_metrics.csv"

    headers = [
        "dataset",
        "best_rpa_f1",
        "best_rpa_precision",
        "best_rpa_recall",
        "best_rpa_threshold",
        "best_rpa_epoch",
        "best_pa_f1",
        "best_pa_precision",
        "best_pa_recall",
        "best_pa_threshold",
        "best_pa_epoch",
        "best_pw_f1",
        "best_pw_precision",
        "best_pw_recall",
        "best_pw_threshold",
        "best_pw_epoch",
        "best_aff_f1",
        "best_aff_precision",
        "best_aff_recall",
        "best_aff_threshold",
        "best_aff_epoch",
    ]

    row = {
        "dataset": dataset_name,
        "best_rpa_f1": metrics.get("best_rpa_f1", ""),
        "best_rpa_precision": metrics.get("best_rpa_precision", ""),
        "best_rpa_recall": metrics.get("best_rpa_recall", ""),
        "best_rpa_threshold": metrics.get("best_rpa_threshold", ""),
        "best_rpa_epoch": metrics.get("best_rpa_epoch", ""),
        "best_pa_f1": metrics.get("best_pa_f1", ""),
        "best_pa_precision": metrics.get("best_pa_precision", ""),
        "best_pa_recall": metrics.get("best_pa_recall", ""),
        "best_pa_threshold": metrics.get("best_pa_threshold", ""),
        "best_pa_epoch": metrics.get("best_pa_epoch", ""),
        "best_pw_f1": metrics.get("best_pw_f1", ""),
        "best_pw_precision": metrics.get("best_pw_precision", ""),
        "best_pw_recall": metrics.get("best_pw_recall", ""),
        "best_pw_threshold": metrics.get("best_pw_threshold", ""),
        "best_pw_epoch": metrics.get("best_pw_epoch", ""),
        "best_aff_f1": metrics.get("best_aff_f1", ""),
        "best_aff_precision": metrics.get("best_aff_precision", ""),
        "best_aff_recall": metrics.get("best_aff_recall", ""),
        "best_aff_threshold": metrics.get("best_aff_threshold", ""),
        "best_aff_epoch": metrics.get("best_aff_epoch", ""),
    }

    file_exists = log_path.exists()
    with log_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive Swift Hydra training for KPI dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="KPI_npz/01_05f10d3a_239c_3bef_9bdc_a2feeb0037aa.npz",
        help="Path to KPI .npz dataset",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of adaptive episodes")
    parser.add_argument("--gen-windows", type=int, default=32, help="Synthetic windows per episode")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for loaders")
    parser.add_argument("--vae-epochs", type=int, default=5, help="TCN-VAE epochs per episode")
    parser.add_argument("--detector-epochs", type=int, default=3, help="Detector epochs per episode")
    parser.add_argument("--final-epochs", type=int, default=60, help="Final detector epochs")
    parser.add_argument("--window-size", type=int, default=None, help="Override window size (requires matching pretrain)")
    parser.add_argument("--window-stride", type=int, default=None, help="Override window stride")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for perturbation loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for zero perturbation loss")
    parser.add_argument("--zeta", type=float, default=1.0, help="Weight for enhanced KL loss")
    parser.add_argument("--delta-min", type=float, default=0.1, help="Triplet margin lower bound")
    parser.add_argument("--delta-max", type=float, default=1.0, help="Triplet margin upper bound")
    parser.add_argument("--sigma-prior", type=float, default=0.5, help="Prior std for enhanced KL")
    parser.add_argument("--loss-weight", type=float, default=0.1, help="Weight for total loss in adversarial objective")
    parser.add_argument("--adv-steps", type=int, default=25, help="Latent optimization steps")
    parser.add_argument("--adv-lr", type=float, default=0.01, help="Latent optimization learning rate")
    return parser.parse_args()


def main(
    dataset_path: str,
    num_episodes: int = 50,
    num_gen_windows: int = 32,
    batch_size: int = 64,
    vae_epochs_per_episode: int = 5,
    detector_epochs_per_episode: int = 3,
    num_epochs_final: int = 60,
    alpha_recon: float = 1.0,
    beta_perturb: float = 1.0,
    gamma_zero: float = 1.0,
    zeta_en_kl: float = 1.0,
    delta_min: float = 0.1,
    delta_max: float = 1.0,
    sigma_prior: float = 0.5,
    total_loss_weight: float = 0.1,
    adv_steps: int = 25,
    adv_lr: float = 0.01,
    window_size_override: int | None = None,
    window_stride_override: int | None = None,
):
    dataset_path = str(dataset_path)
    dataset_name = Path(dataset_path).stem
    save_dir = Path("./saved_models") / dataset_name
    vae_path = save_dir / "tcn_vae.pth"
    detector_path = save_dir / "tcn_transformer_detector.pth"
    meta_path = save_dir / "tcn_training_meta.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_np, y_train_np, X_test_np, y_test_np = load_dataset(dataset_path)
    X_train_np = ensure_2d(X_train_np)
    X_test_np = ensure_2d(X_test_np)

    y_train_np = y_train_np.reshape(-1)
    y_test_np = y_test_np.reshape(-1)

    if os.path.exists(meta_path):
        meta = torch.load(meta_path, weights_only=False)
        scaler = build_scaler_from_meta(meta)
        meta_window_size = int(meta["window_size"])
        meta_window_stride = int(meta["window_stride"])
        if window_size_override is not None and window_size_override != meta_window_size:
            raise ValueError(
                "Window size override does not match pretraining metadata. "
                "Re-run pretrain_tcn.py with the desired window size or remove the saved model directory."
            )
        if window_stride_override is not None and window_stride_override != meta_window_stride:
            raise ValueError(
                "Window stride override does not match pretraining metadata. "
                "Re-run pretrain_tcn.py with the desired stride or remove the saved model directory."
            )
        window_size = window_size_override or meta_window_size
        window_stride = window_stride_override or meta_window_stride
        input_channels = int(meta["input_channels"])
        detector_input_size = int(meta["detector_input_size"])
    else:
        scaler = StandardScaler()
        scaler.fit(X_train_np)
        window_size = window_size_override or 128
        window_stride = window_stride_override or 32
        input_channels = X_train_np.shape[1]
        detector_input_size = input_channels * window_size

    X_train_scaled = scaler.transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)

    window_processor = WindowDataProcessor(window_size=window_size, stride=window_stride)

    train_series = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(0)
    train_labels_seq = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(0)
    train_windows, train_window_labels = window_processor.create_windows(train_series, train_labels_seq)

    test_series = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(0)
    test_labels_seq = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(0)
    test_windows, test_window_labels = window_processor.create_windows(test_series, test_labels_seq)

    if train_windows.size(0) == 0 or test_windows.size(0) == 0:
        raise RuntimeError("Windowing produced empty datasets. Check window size/stride configuration.")

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

    online_detector = TransformerDetector(input_size=detector_input_size).to(device)

    if os.path.exists(vae_path):
        tcn_vae.load_state_dict(torch.load(vae_path, map_location=device))
    if os.path.exists(detector_path):
        state_dict = torch.load(detector_path, map_location=device)
        online_detector.load_state_dict(state_dict)

    current_windows = train_windows.clone()
    current_labels = train_window_labels.clone()

    flattened_test = test_windows.reshape(test_windows.size(0), -1)
    test_dataset = TensorDataset(flattened_test, test_window_labels.squeeze(1))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    merlion_evaluator = MerlionEvaluator(verbose=False)

    optimizer_vae = Adam(tcn_vae.parameters(), lr=1e-4)
    optimizer_detector = Adam(online_detector.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    latent_dim = tcn_vae.latent_dim
    state_dim = latent_dim * 2
    policy_hidden = max(256, state_dim)
    policy_net = PolicyNetwork(input_dim=state_dim, hidden_dim=policy_hidden, output_dim=latent_dim)
    value_net = ValueNetwork(input_dim=state_dim, hidden_dim=policy_hidden)
    ppo_trainer = PPOTrainer(
        policy_net=policy_net,
        value_net=value_net,
        policy_lr=1e-4,
        value_lr=1e-4,
        gamma=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.001,
        action_std=0.2,
        device=device,
    )

    for episode in range(num_episodes):
        print(f"===== EPISODE {episode + 1}/{num_episodes} =====")

        vae_dataset = TensorDataset(current_windows, current_labels)
        vae_loader = DataLoader(vae_dataset, batch_size=batch_size, shuffle=True)

        # Stage 1: refine TCN-VAE on current dataset
        for epoch_idx in range(vae_epochs_per_episode):
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
                    zeta=zeta_en_kl,
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

            denom = max(len(vae_loader), 1)
            avg_loss = total_loss / denom
            avg_recon = total_recon / denom
            avg_perturb = total_perturb / denom
            avg_zero = total_zero / denom
            avg_kl = total_kl / denom

            if (epoch_idx + 1) % 10 == 0 or epoch_idx == vae_epochs_per_episode - 1:
                print(
                    f"    [TCN-VAE inner] Epoch {epoch_idx + 1}/{vae_epochs_per_episode}, "
                    f"total={avg_loss:.4f}, recon={avg_recon:.4f}, "
                    f"perturb={avg_perturb:.4f}, zero={avg_zero:.4f}, kl={avg_kl:.4f}"
                )

        # Build detector loader on current windows
        flattened_features = current_windows.reshape(current_windows.size(0), -1)
        detector_dataset = TensorDataset(flattened_features, current_labels.squeeze(1))
        detector_loader = DataLoader(detector_dataset, batch_size=batch_size, shuffle=True)

        # Stage 2: train detector prior to generation
        for _ in range(detector_epochs_per_episode):
            train_loss = train_detector(online_detector, detector_loader, optimizer_detector, criterion, device)
        print(f"  Detector training loss (pre-augmentation): {train_loss:.4f}")

        baseline_rpa, baseline_pa, _ = evaluate_detector_merlion(online_detector, test_loader, device, merlion_evaluator)
        print(f"  Baseline metrics -> RPA F1: {baseline_rpa:.4f}, PA F1: {baseline_pa:.4f}")

        labels_flat = current_labels.squeeze(1)
        idx_class1 = (labels_flat == 1).nonzero(as_tuple=True)[0]
        idx_class0 = (labels_flat == 0).nonzero(as_tuple=True)[0]

        if len(idx_class1) == 0 or len(idx_class0) == 0:
            print("  Skipping adversarial generation due to class imbalance.")
            continue

        generated_windows: List[torch.Tensor] = []
        generated_labels: List[torch.Tensor] = []
        episode_states: List[torch.Tensor] = []
        episode_actions: List[torch.Tensor] = []
        episode_log_probs: List[torch.Tensor] = []

        for _ in range(num_gen_windows):
            random_idx = random.choice(idx_class1)
            base_window = current_windows[random_idx]
            y_target = torch.full((1, 1), 0.8, device=device)
            with torch.no_grad():
                mu, logvar = tcn_vae.encode(base_window.unsqueeze(0).to(device), y_target)

            mu = mu.squeeze(0)
            logvar = logvar.squeeze(0)
            state_vec = torch.cat([mu, logvar], dim=-1).detach()
            action, log_prob = ppo_trainer.get_action_and_log_prob(state_vec.unsqueeze(0))
            action_vec = action.squeeze(0)
            log_prob_vec = log_prob.squeeze(0)

            std = torch.exp(0.5 * logvar)
            delta = torch.tanh(action_vec)
            z = mu + delta * std

            with torch.no_grad():
                adv_window = tcn_vae.decode(z.unsqueeze(0), y_target).detach().cpu().squeeze(0)

            generated_windows.append(adv_window.unsqueeze(0))
            generated_labels.append(torch.ones(1, 1))

            episode_states.append(state_vec.detach())
            episode_actions.append(action_vec.detach())
            episode_log_probs.append(log_prob_vec.detach())

        if generated_windows:
            new_windows = torch.cat(generated_windows, dim=0)
            new_labels = torch.cat(generated_labels, dim=0)
            current_windows = torch.cat([current_windows, new_windows], dim=0)
            current_labels = torch.cat([current_labels, new_labels], dim=0)

            # Retrain detector on augmented dataset
            flattened_features = current_windows.reshape(current_windows.size(0), -1)
            detector_dataset = TensorDataset(flattened_features, current_labels.squeeze(1))
            detector_loader = DataLoader(detector_dataset, batch_size=batch_size, shuffle=True)
            for _ in range(detector_epochs_per_episode):
                train_loss = train_detector(online_detector, detector_loader, optimizer_detector, criterion, device)
            print(f"  Detector training loss (post-augmentation): {train_loss:.4f}")

            new_rpa, new_pa, _ = evaluate_detector_merlion(online_detector, test_loader, device, merlion_evaluator)
            reward = (new_rpa - baseline_rpa) + (new_pa - baseline_pa)
            print(
                f"  Updated metrics -> RPA F1: {new_rpa:.4f}, PA F1: {new_pa:.4f}, "
                f"Reward: {reward:.4f}"
            )

            if episode_states:
                states_tensor = torch.stack(episode_states).to(device)
                actions_tensor = torch.stack(episode_actions).to(device)
                old_log_probs_tensor = torch.stack(episode_log_probs).unsqueeze(-1).to(device)
                rewards_tensor = torch.full((states_tensor.size(0), 1), reward, device=device)
                values = ppo_trainer.value_net(states_tensor).detach()
                next_values = torch.zeros_like(values)
                dones = torch.ones_like(rewards_tensor)
                advantages = ppo_trainer.compute_advantages(rewards_tensor, values, next_values, dones)
                returns = values + advantages

                ppo_trainer.policy_net.train()
                ppo_trainer.value_net.train()
                ppo_trainer.ppo_update(
                    states_tensor,
                    actions_tensor,
                    old_log_probs_tensor,
                    returns.detach(),
                    advantages.detach(),
                )
        else:
            print("  No synthetic windows generated this episode.")

    # Final detector training on augmented dataset
    flattened_features = current_windows.reshape(current_windows.size(0), -1)
    final_dataset = TensorDataset(flattened_features, current_labels.squeeze(1))
    final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True)

    test_features = test_windows.reshape(test_windows.size(0), -1)
    test_dataset = TensorDataset(test_features, test_window_labels.squeeze(1))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    final_detector = TransformerDetector(input_size=flattened_features.size(1)).to(device)
    final_optimizer = Adam(final_detector.parameters(), lr=1e-3)

    best_rpa = 0.0
    best_pa = 0.0
    best_pw = 0.0
    best_aff_f1 = 0.0

    best_rpa_epoch = 0
    best_pa_epoch = 0
    best_pw_epoch = 0
    best_aff_epoch = 0
    
    best_rpa_metrics: dict[str, float] | None = None
    best_pa_metrics: dict[str, float] | None = None
    best_pw_metrics: dict[str, float] | None = None
    best_aff_metrics: dict[str, float] | None = None

    for epoch in range(num_epochs_final):
        train_loss = train_detector(final_detector, final_loader, final_optimizer, criterion, device)
        print(f"[{dataset_name}] [Final Detector] Epoch {epoch + 1}/{num_epochs_final}, Loss={train_loss:.4f}")

        evaluate_with_classification_report_and_auc(final_detector, test_loader, device, threshold=0.5)

        with torch.no_grad():
            score_batches = []
            label_batches = []
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = final_detector(X_batch).view(-1)
                score_batches.append(preds.detach().cpu())
                label_batches.append(y_batch.detach().cpu().view(-1))

        scores = torch.cat(score_batches).numpy() if score_batches else np.array([])
        labels = torch.cat(label_batches).numpy() if label_batches else np.array([])

        if scores.size and labels.size:
            merlion_result = merlion_evaluator.evaluate_comprehensive(scores, labels, optimization_method="floating")
            search_details = getattr(merlion_evaluator, "_last_search", None) or {}
            thresholds = search_details.get("thresholds", {})

            rpa_score_obj = search_details.get("rpa_score_max")
            pa_score_obj = search_details.get("pa_score_max")
            pw_score_obj = search_details.get("pw_score_max")
            aff_dict = search_details.get("affiliation_max", {}) or {}

            def _safe_metric(score_obj, score_type):
                if score_obj is None:
                    return 0.0, 0.0, 0.0
                precision = float(score_obj.precision(score_type))
                recall = float(score_obj.recall(score_type))
                f1 = float(score_obj.f1(score_type))
                return precision, recall, f1

            rpa_precision, rpa_recall, current_rpa = _safe_metric(rpa_score_obj, ScoreType.RevisedPointAdjusted)
            pa_precision, pa_recall, current_pa = _safe_metric(pa_score_obj, ScoreType.PointAdjusted)
            pw_precision, pw_recall, current_pw = _safe_metric(pw_score_obj, ScoreType.Pointwise)

            aff_precision = float(aff_dict.get("precision", 0.0))
            aff_recall = float(aff_dict.get("recall", 0.0))
            aff_f1 = 0.0
            if aff_precision + aff_recall > 0:
                aff_f1 = 2 * aff_precision * aff_recall / (aff_precision + aff_recall)

            print(
                "[Merlion] "
                f"RPA F1={current_rpa:.4f} (P={rpa_precision:.4f}, R={rpa_recall:.4f}), "
                f"PA F1={current_pa:.4f} (P={pa_precision:.4f}, R={pa_recall:.4f}), "
                f"PW F1={current_pw:.4f}, "
                f"Aff F1={aff_f1:.4f}"
            )

            if aff_f1 > best_aff_f1:
                best_aff_f1 = aff_f1
                best_aff_epoch = epoch + 1
                best_aff_metrics = {
                    "precision": aff_precision,
                    "recall": aff_recall,
                    "f1": aff_f1,
                    "threshold": float(thresholds.get("affiliation", 0.0)),
                    "epoch": epoch + 1,
                }

            if current_rpa > best_rpa:
                best_rpa = current_rpa
                best_rpa_epoch = epoch + 1
                best_rpa_metrics = {
                    "precision": rpa_precision,
                    "recall": rpa_recall,
                    "f1": current_rpa,
                    "threshold": float(thresholds.get("rpa", 0.0)),
                    "epoch": epoch + 1,
                    "anomaly_count": int(merlion_result.get("anomaly_count", labels.sum())),
                    "predicted_anomaly_count": int(merlion_result.get("predicted_anomaly_count", 0)),
                }

            if current_pa > best_pa:
                best_pa = current_pa
                best_pa_epoch = epoch + 1
                best_pa_metrics = {
                    "precision": pa_precision,
                    "recall": pa_recall,
                    "f1": current_pa,
                    "threshold": float(thresholds.get("pa", 0.0)),
                    "epoch": epoch + 1,
                }

            if current_pw > best_pw:
                best_pw = current_pw
                best_pw_epoch = epoch + 1
                best_pw_metrics = {
                    "precision": pw_precision,
                    "recall": pw_recall,
                    "f1": current_pw,
                    "threshold": float(thresholds.get("pw", 0.0)),
                    "epoch": epoch + 1,
                }

        print("-" * 40)

    print(f"[{dataset_name}] Best Merlion metrics (TCN pipeline):")

    if best_aff_metrics:
        print(
            "  Affiliation metrics: "
            f"Precision={best_aff_metrics['precision']:.4f}, "
            f"Recall={best_aff_metrics['recall']:.4f}, "
            f"F1={best_aff_metrics['f1']:.4f}, "
            f"threshold={best_aff_metrics['threshold']:.4f}, epoch={best_aff_metrics['epoch']}"
        )
    else:
        print("  Affiliation metrics: No anomalies detected or metrics unavailable")

    if best_rpa_metrics:
        print(
            "  RPA (Revised Point-Adjusted): "
            f"Precision={best_rpa_metrics['precision']:.4f}, "
            f"Recall={best_rpa_metrics['recall']:.4f}, "
            f"F1={best_rpa_metrics['f1']:.4f}, "
            f"threshold={best_rpa_metrics['threshold']:.4f}, epoch={best_rpa_metrics['epoch']}"
        )
    else:
        print("  RPA (Revised Point-Adjusted): No improvement recorded")

    if best_pa_metrics:
        print(
            "  Point-Adjusted metrics: "
            f"Precision={best_pa_metrics['precision']:.4f}, "
            f"Recall={best_pa_metrics['recall']:.4f}, "
            f"F1={best_pa_metrics['f1']:.4f}, "
            f"threshold={best_pa_metrics['threshold']:.4f}, epoch={best_pa_metrics['epoch']}"
        )
    else:
        print("  Point-Adjusted metrics: No improvement recorded")

    if best_pw_metrics:
        print(
            "  Point-wise metrics: "
            f"Precision={best_pw_metrics['precision']:.4f}, "
            f"Recall={best_pw_metrics['recall']:.4f}, "
            f"F1={best_pw_metrics['f1']:.4f}, "
            f"threshold={best_pw_metrics['threshold']:.4f}, epoch={best_pw_metrics['epoch']}"
        )
    else:
        print("  Point-wise metrics: No improvement recorded")

    metrics_payload = {
        "best_rpa_f1": best_rpa_metrics.get("f1") if best_rpa_metrics else "",
        "best_rpa_precision": best_rpa_metrics.get("precision") if best_rpa_metrics else "",
        "best_rpa_recall": best_rpa_metrics.get("recall") if best_rpa_metrics else "",
        "best_rpa_threshold": best_rpa_metrics.get("threshold") if best_rpa_metrics else "",
        "best_rpa_epoch": best_rpa_metrics.get("epoch") if best_rpa_metrics else "",
        "best_pa_f1": best_pa_metrics.get("f1") if best_pa_metrics else "",
        "best_pa_precision": best_pa_metrics.get("precision") if best_pa_metrics else "",
        "best_pa_recall": best_pa_metrics.get("recall") if best_pa_metrics else "",
        "best_pa_threshold": best_pa_metrics.get("threshold") if best_pa_metrics else "",
        "best_pa_epoch": best_pa_metrics.get("epoch") if best_pa_metrics else "",
        "best_pw_f1": best_pw_metrics.get("f1") if best_pw_metrics else "",
        "best_pw_precision": best_pw_metrics.get("precision") if best_pw_metrics else "",
        "best_pw_recall": best_pw_metrics.get("recall") if best_pw_metrics else "",
        "best_pw_threshold": best_pw_metrics.get("threshold") if best_pw_metrics else "",
        "best_pw_epoch": best_pw_metrics.get("epoch") if best_pw_metrics else "",
        "best_aff_f1": best_aff_metrics.get("f1") if best_aff_metrics else "",
        "best_aff_precision": best_aff_metrics.get("precision") if best_aff_metrics else "",
        "best_aff_recall": best_aff_metrics.get("recall") if best_aff_metrics else "",
        "best_aff_threshold": best_aff_metrics.get("threshold") if best_aff_metrics else "",
        "best_aff_epoch": best_aff_metrics.get("epoch") if best_aff_metrics else "",
    }

    log_best_metrics(dataset_name, metrics_payload)

    summary_dir = Path("./results")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"SwiftHydra_tcn_{dataset_name}_summary.csv"

    if any(metric is not None for metric in (best_aff_metrics, best_rpa_metrics, best_pa_metrics, best_pw_metrics)):
        if summary_path.exists():
            df = pd.read_csv(summary_path, index_col=0)
        else:
            df = pd.DataFrame()

        column_name = f"SwiftHydraTCN_{df.shape[1]}"

        df.loc["Dataset", column_name] = dataset_name

        def assign_metric(label: str, metrics: dict[str, float] | None, key: str, digits: int = 5):
            if metrics is None:
                df.loc[label, column_name] = np.nan
            else:
                value = metrics.get(key)
                if value is None:
                    df.loc[label, column_name] = np.nan
                else:
                    if digits == 0:
                        df.loc[label, column_name] = int(value)
                    else:
                        df.loc[label, column_name] = round(float(value), digits)

        assign_metric("Affiliation Precision", best_aff_metrics, "precision")
        assign_metric("Affiliation Recall", best_aff_metrics, "recall")
        assign_metric("Affiliation F1", best_aff_metrics, "f1")
        assign_metric("Affiliation Threshold", best_aff_metrics, "threshold", digits=6)
        assign_metric("Affiliation Epoch", best_aff_metrics, "epoch", digits=0)

        assign_metric("RPA Precision", best_rpa_metrics, "precision")
        assign_metric("RPA Recall", best_rpa_metrics, "recall")
        assign_metric("RPA F1", best_rpa_metrics, "f1")
        assign_metric("RPA Threshold", best_rpa_metrics, "threshold", digits=6)
        assign_metric("RPA Epoch", best_rpa_metrics, "epoch", digits=0)
        assign_metric("RPA True Anomalies", best_rpa_metrics, "anomaly_count", digits=0)
        assign_metric("RPA Predicted Anomalies", best_rpa_metrics, "predicted_anomaly_count", digits=0)

        assign_metric("PA Precision", best_pa_metrics, "precision")
        assign_metric("PA Recall", best_pa_metrics, "recall")
        assign_metric("PA F1", best_pa_metrics, "f1")
        assign_metric("PA Threshold", best_pa_metrics, "threshold", digits=6)
        assign_metric("PA Epoch", best_pa_metrics, "epoch", digits=0)

        assign_metric("Point-wise Precision", best_pw_metrics, "precision")
        assign_metric("Point-wise Recall", best_pw_metrics, "recall")
        assign_metric("Point-wise F1", best_pw_metrics, "f1")
        assign_metric("Point-wise Threshold", best_pw_metrics, "threshold", digits=6)
        assign_metric("Point-wise Epoch", best_pw_metrics, "epoch", digits=0)

        df.to_csv(summary_path, index=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset,
        num_episodes=args.episodes,
        num_gen_windows=args.gen_windows,
        batch_size=args.batch_size,
        vae_epochs_per_episode=args.vae_epochs,
        detector_epochs_per_episode=args.detector_epochs,
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
        window_size_override=args.window_size,
        window_stride_override=args.window_stride,
    )
