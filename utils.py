import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import math

# =========================
# 1. Load & Utility Functions
# =========================

def load_adbench_data(dataset_path):
    """
    Load dataset from a .npz file.
    Assumes the file contains:
    - 'X': Feature matrix (N, d)
    - 'y': Labels (N,)
    """
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def evaluate_with_classification_report_and_auc(model, test_loader, device, threshold=0.5):
    """
    Evaluate a model using classification report and AUC-ROC metric.
    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for test data.
        device: Computation device (CPU/GPU).
        threshold: Threshold for binary classification.
    Returns:
        Classification report and AUC-ROC score.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()  # Predicted scores (B,)
            all_preds.append(y_pred.cpu())
            all_labels.append(y_batch.cpu())

    preds = torch.cat(all_preds).numpy()  # Flatten predictions
    labels = torch.cat(all_labels).numpy()  # Flatten labels

    # Convert predictions to binary labels
    binary_preds = (preds > threshold).astype(int)

    # Generate classification report
    report = classification_report(labels, binary_preds, target_names=['Class 0', 'Class 1'])
    print(report)

    # Calculate AUC-ROC if both classes are present
    if len(set(labels)) > 1:
        aucroc = roc_auc_score(labels, preds)
        print(f"AUC-ROC: {aucroc:.4f}")
    else:
        aucroc = None
        print("AUC-ROC: Undefined (only one class present in labels)")

    return report, aucroc

def log_to_file(file_path, message):
    """
    Append a log message to the specified file.
    Args:
        file_path: Path to the log file.
        message: Message to log.
    """
    with open(file_path, "a") as file:
        file.write(message + "\n")

def beta_cvae_loss_fn(x, x_recon, mean, logvar, beta=4.0):
    """
    Compute Beta-CVAE loss (Reconstruction + Beta * KL Divergence).
    Args:
        x: Original input data.
        x_recon: Reconstructed data.
        mean: Mean of latent space distribution.
        logvar: Log variance of latent space distribution.
        beta: Weight for KL divergence.
    Returns:
        Total loss (scalar).
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def train_beta_cvae(model, data_loader, optimizer, device):
    """
    Train Beta-CVAE model for one epoch.
    Args:
        model: Beta-CVAE model to train.
        data_loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        device: Computation device (CPU/GPU).
    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)  # Reshape labels (B, 1)

        # Forward pass
        x_recon, mean, logvar = model(x_batch, y_batch)
        loss = beta_cvae_loss_fn(x_batch, x_recon, mean, logvar, beta=model.beta)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def train_detector(model, train_loader, optimizer, criterion, device):
    """
    Train a detector model for one epoch.
    Args:
        model: Detector model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        criterion: Loss function (e.g., BCE Loss).
        device: Computation device (CPU/GPU).
    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


# =========================
# Perturbation Loss Components
# =========================

def mse_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared distance between tensors a and b."""
    if a.shape != b.shape:
        raise ValueError("Inputs to mse_distance must share the same shape")
    if a.dim() == 1:
        return torch.square(a - b)
    reduction_dims = tuple(range(1, a.dim()))
    return torch.mean(torch.square(a - b), dim=reduction_dims)


def perturbation_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    x_tilde: torch.Tensor,
    delta_min: float = 0.1,
    delta_max: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Triplet-style perturbation loss encouraging synthetic anomalies to be distinct yet realistic.

    Args:
        x: Original normal samples (B, ...).
        x_recon: Reconstructions of x (B, ...).
        x_tilde: Synthetic anomalous samples (B, ...).
        delta_min: Minimum separation margin between reconstructions and anomalies.
        delta_max: Maximum allowed distance for anomalies to stay plausible.
        reduction: "mean", "sum", or "none" for per-batch aggregation.
    """
    if not (x.shape == x_recon.shape == x_tilde.shape):
        raise ValueError("All inputs to perturbation_loss must share the same shape")

    dist_pos = mse_distance(x, x_recon)
    dist_neg = mse_distance(x, x_tilde)

    triplet_term = torch.clamp(dist_pos - dist_neg + delta_min, min=0.0)
    regularization_term = torch.clamp(dist_neg - delta_max, min=0.0)
    loss = triplet_term + regularization_term

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


# =========================
# Zero-value Perturbation Loss
# =========================

def zero_perturbation_loss(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Encourage meaningful deviations on zero-valued dimensions of x."""
    if x.shape != x_tilde.shape:
        raise ValueError("Inputs to zero_perturbation_loss must share the same shape")

    flat_x = x.view(x.size(0), -1)
    flat_tilde = x_tilde.view(x_tilde.size(0), -1)
    mask = (flat_x == 0.0).float()
    zero_counts = mask.sum(dim=1)

    per_element_dist = torch.square(flat_x - flat_tilde)
    masked_dist = per_element_dist * mask

    seq_length = x.size(1) if x.dim() > 1 else 1

    eps = 1e-8
    normalized = masked_dist.sum(dim=1) / (seq_length * (zero_counts + eps))
    loss = torch.pow(normalized + 1.0, -1)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


# =========================
# Enhanced KL Divergence Loss
# =========================

def enhanced_kl_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    sigma_prior: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL regularizer against tightened Gaussian prior N(0, sigma_prior^2)."""
    if sigma_prior <= 0:
        raise ValueError("sigma_prior must be positive")
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must share the same shape")

    var = torch.exp(logvar)
    prior_var = sigma_prior ** 2
    log_sigma_prior = math.log(sigma_prior)

    elements = 1 + logvar - mu.pow(2) - (var / prior_var) + 2 * log_sigma_prior
    per_sample = -0.5 * torch.sum(elements, dim=1)

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def total_anomaly_vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    x_tilde: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    zeta: float = 1.0,
    delta_min: float = 0.1,
    delta_max: float = 1.0,
    sigma_prior: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine reconstruction, perturbation, zero-perturbation, and enhanced KL losses."""
    recon = F.mse_loss(x_recon, x, reduction="mean")
    total = alpha * recon

    if x_tilde is not None:
        perturb = perturbation_loss(
            x=x,
            x_recon=x_recon,
            x_tilde=x_tilde,
            delta_min=delta_min,
            delta_max=delta_max,
            reduction="mean",
        )
        zero = zero_perturbation_loss(x=x, x_tilde=x_tilde, reduction="mean")
        total = total + beta * perturb + gamma * zero
    else:
        device = x.device if isinstance(x, torch.Tensor) else "cpu"
        perturb = torch.tensor(0.0, device=device)
        zero = torch.tensor(0.0, device=device)

    kl = enhanced_kl_loss(mu=mu, logvar=logvar, sigma_prior=sigma_prior, reduction="mean")
    total = total + zeta * kl

    return total, recon, perturb, zero, kl


# Để tái lập trình ngẫu nhiên cho ví dụ
torch.manual_seed(0)
np.random.seed(0)

# Giả sử ta có một hàm tính reward liên quan đến "độ đa dạng" (diversity)
# Ở đây, tạm thời ta giả lập bằng cách random ra reward để minh hoạ.
def compute_diversity_reward(modified_z):
    # Tùy chỉnh cách tính reward thực tế.
    # Ở đây minh hoạ: reward tỉ lệ với độ lớn L2 norm của z (giả sử).
    return torch.norm(modified_z, p=2, dim=-1, keepdim=True)

# Hàm tiện ích chuyển numpy -> torch
def to_tensor(x, device="cpu", dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=dtype)

def One_Step_To_Feasible_Action(
        beta_cvae,
        detector,
        x_orig,
        device,
        previously_generated=None,
        alpha=1.0,
        lambda_div=0.1,
        lr=0.001,
        steps=50,
        log_file=None
):
    """
    Generate adversarial samples by modifying latent space representation.
    Args:
        beta_cvae: Trained Beta-CVAE model.
        detector: Trained detector model.
        x_orig: Original input data.
        device: Computation device (CPU/GPU).
        previously_generated: List of previously generated samples (for diversity).
        alpha: Scaling factor for diversity term.
        lambda_div: Weight for diversity term.
        lr: Learning rate for optimization.
        steps: Number of optimization steps.
        log_file: Path to log file for recording progress.
    Returns:
        Adversarial sample (torch.Tensor).
    """
    beta_cvae.eval()
    detector.eval()

    if previously_generated is None:
        previously_generated = []

    x_orig = x_orig.to(device).unsqueeze(0)  # Reshape to batch format (1, d)
    y_class1 = torch.full((1, 1), 0.8, device=device)  # Target class label (e.g., 0.8)

    # Encode input data into latent space
    with torch.no_grad():
        mean, logvar = beta_cvae.encode(x_orig, y_class1)

    mu = mean.detach()
    sigma = torch.exp(0.5 * logvar).detach()
    epsilon = torch.randn_like(sigma)

    psi_param = torch.zeros_like(sigma, requires_grad=True)
    optimizer_psi = torch.optim.Adam([psi_param], lr=lr)

    for step in range(steps):
        optimizer_psi.zero_grad()

        psi = torch.exp(psi_param)
        z = mu + psi * (sigma * epsilon)

        x_synthetic = beta_cvae.decode(z, y_class1)

        # Calculate detector prediction
        prob_class1 = detector(x_synthetic)

        # Diversity term (if previous samples exist)
        if previously_generated:
            x_old_cat = torch.stack(previously_generated, dim=0).to(device)  # Stack previous samples (N, d)
            dist = torch.norm(x_synthetic - x_old_cat, p=2, dim=1)  # Pairwise distances
            diversity_term = torch.exp(-alpha * dist).sum()
        else:
            diversity_term = 0.0

        # Calculate total reward (inverse objective)
        inv_reward = prob_class1.mean() + lambda_div * diversity_term
        inv_reward.backward()
        optimizer_psi.step()

    psi = torch.exp(psi_param).detach()
    z_final = mu + psi * (sigma * epsilon)

    diversity_value = (
        diversity_term.detach().item() if isinstance(diversity_term, torch.Tensor) else float(diversity_term)
    )

    print(
        f"Deceiving Detector Reward: {1 / (prob_class1.item() + 1e-4):.4f}",
        f"Diversity reward: {1 / (diversity_value + 1e-4):.4f}",
        f"Sample reward: {1 / (inv_reward.item() + 1e-4):.4f}",
        f"Psi mean: {psi.mean().item():.4f}"
    )

    # Decode optimized latent variable back to data space
    with torch.no_grad():
        x_adv = beta_cvae.decode(z_final, y_class1).detach().cpu().squeeze(0)
    return x_adv
