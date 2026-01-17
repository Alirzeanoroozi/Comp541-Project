# trainer_utils.py
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


# loss function for entropy regularization fusion mil
def attn_entropy(alpha, eps=1e-8):
    try:
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.clamp_min(eps)
            return -(alpha * alpha.log()).sum(dim=1).mean()
    except Exception:
        pass

    # Fallback to numpy
    alpha = np.asarray(alpha)
    alpha = np.clip(alpha, eps, None)
    return float((-(alpha * np.log(alpha)).sum(axis=1)).mean())

# metrics utils
def regression_metrics(targets, preds):
    targets = np.asarray(targets, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.float32)
    mse = mean_squared_error(targets, preds)

    rho = spearmanr(targets, preds).correlation
    if not np.isfinite(rho):
        rho = 0.0

    return float(mse), float(rho)


def best_min(values):
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    idx = int(np.nanargmin(arr))
    return float(arr[idx]), idx + 1


def best_max(values):
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    idx = int(np.nanargmax(arr))
    return float(arr[idx]), idx + 1

# plotting utilities
def plot_training_curves(history, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if len(list(epochs)) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    # loss
    axes[0].plot(epochs, history.get("train_loss", []), label="Train Loss")
    axes[0].plot(epochs, history.get("val_loss", []), label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # mse
    axes[1].plot(epochs, history.get("train_mse", []), label="Train MSE")
    axes[1].plot(epochs, history.get("val_mse", []), label="Val MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Training and Validation MSE")
    axes[1].legend()
    axes[1].grid(True)

    # spearman corr
    axes[2].plot(epochs, history.get("train_spearman", []), label="Train Spearman")
    axes[2].plot(epochs, history.get("val_spearman", []), label="Val Spearman")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Spearman")
    axes[2].set_title("Training and Validation Spearman")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_predictions(predictions, targets, epoch, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # scatter
    axes[0].scatter(targets, predictions, alpha=0.5, s=10)
    min_val = float(min(targets.min(), predictions.min()))
    max_val = float(max(targets.max(), predictions.max()))
    axes[0].plot([min_val, max_val], [min_val, max_val], linestyle="--", lw=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title(f"Predictions vs Actual (Epoch {epoch})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # residuals
    residuals = predictions - targets
    axes[1].scatter(targets, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, linestyle="--", lw=2)
    axes[1].set_xlabel("Actual Values")
    axes[1].set_ylabel("Residuals (Predicted - Actual)")
    axes[1].set_title(f"Residual Plot (Epoch {epoch})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    epoch_str = f"epoch_{epoch}" if isinstance(epoch, int) else str(epoch)
    plt.savefig(save_dir / f"predictions_{epoch_str}.png", dpi=300, bbox_inches="tight")
    plt.close()

# save utilities
def save_predictions_csv(predictions, targets, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"predictions": predictions, "targets": targets})
    df.to_csv(save_path, index=False)


def update_metrics_summary_json(history, run_metrics: dict, summary_path):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    best_train_loss, best_train_loss_ep = best_min(history.get("train_loss", []))
    best_val_loss, best_val_loss_ep = best_min(history.get("val_loss", []))
    best_train_mse, best_train_mse_ep = best_min(history.get("train_mse", []))
    best_val_mse, best_val_mse_ep = best_min(history.get("val_mse", []))
    best_train_spearman, best_train_spearman_ep = best_max(history.get("train_spearman", []))
    best_val_spearman, best_val_spearman_ep = best_max(history.get("val_spearman", []))

    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception:
            summary = {}
    else:
        summary = {}

    summary["best_metrics"] = {
        "train": {
            "loss": {"value": best_train_loss, "epoch": best_train_loss_ep},
            "mse": {"value": best_train_mse, "epoch": best_train_mse_ep},
            "spearman": {"value": best_train_spearman, "epoch": best_train_spearman_ep},
        },
        "val": {
            "loss": {"value": best_val_loss, "epoch": best_val_loss_ep},
            "mse": {"value": best_val_mse, "epoch": best_val_mse_ep},
            "spearman": {"value": best_val_spearman, "epoch": best_val_spearman_ep},
        },
    }

    summary.setdefault("runs", {})
    # caller should choose key like "final" or "test" or int epoch as str
    run_key = str(run_metrics.get("run_key", "run"))
    summary["runs"][run_key] = {k: v for k, v in run_metrics.items() if k != "run_key"}

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
