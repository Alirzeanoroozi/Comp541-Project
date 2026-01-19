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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(history, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = len(history.get("train_loss", []))
    if n_epochs == 0:
        return
    epochs = list(range(1, n_epochs + 1))

    def _safe_plot(ax, key, label):
        y = history.get(key, None)
        if y is None:
            return False
        if not isinstance(y, (list, tuple, np.ndarray)):
            return False
        if len(y) != n_epochs:
            return False
        ax.plot(epochs, y, label=label)
        return True

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # -------------------------
    # 0) Loss (always expected)
    # -------------------------
    _safe_plot(axes[0], "train_loss", "Train Loss")
    _safe_plot(axes[0], "val_loss", "Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # ------------------------------------------------
    # 1) Middle panel: Regression MSE OR Classification Acc
    # ------------------------------------------------
    plotted_mid = False
    # Prefer regression keys if present
    plotted_mid |= _safe_plot(axes[1], "train_mse", "Train MSE")
    plotted_mid |= _safe_plot(axes[1], "val_mse", "Val MSE")
    if plotted_mid:
        axes[1].set_ylabel("MSE")
        axes[1].set_title("Training and Validation MSE")
    else:
        # Fall back to classification accuracy
        plotted_mid |= _safe_plot(axes[1], "train_acc", "Train Acc")
        plotted_mid |= _safe_plot(axes[1], "val_acc", "Val Acc")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")

    axes[1].set_xlabel("Epoch")
    if plotted_mid:
        axes[1].legend()
    axes[1].grid(True)

    # ------------------------------------------------
    # 2) Right panel: Regression Spearman OR Classification F1/AUC
    # ------------------------------------------------
    plotted_right = False
    # Prefer regression spearman if present
    plotted_right |= _safe_plot(axes[2], "train_spearman", "Train Spearman")
    plotted_right |= _safe_plot(axes[2], "val_spearman", "Val Spearman")
    if plotted_right:
        axes[2].set_ylabel("Spearman")
        axes[2].set_title("Training and Validation Spearman")
    else:
        # Fall back to classification F1 / AUC
        plotted_right |= _safe_plot(axes[2], "train_f1", "Train F1")
        plotted_right |= _safe_plot(axes[2], "val_f1", "Val F1")
        # AUC might be nan-filled; still okay to plot if lengths match
        plotted_right |= _safe_plot(axes[2], "train_auc", "Train AUC")
        plotted_right |= _safe_plot(axes[2], "val_auc", "Val AUC")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Training and Validation F1 / AUC")

    axes[2].set_xlabel("Epoch")
    if plotted_right:
        axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_predictions(predictions, targets, epoch, save_dir):
    """
    Works for both:
      - Regression: scatter + residual plot
      - Classification: confusion matrix + per-class prediction counts
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    preds = np.asarray(predictions)
    targs = np.asarray(targets)

    # Flatten if needed
    preds = preds.reshape(-1)
    targs = targs.reshape(-1)

    is_int_targets = np.all(np.isfinite(targs)) and np.all(np.abs(targs - np.round(targs)) < 1e-6)
    unique_t = np.unique(targs.astype(int)) if is_int_targets else np.array([])
    looks_classification = is_int_targets and (unique_t.size > 1) and (unique_t.size <= 50)

    epoch_str = f"epoch_{epoch}" if isinstance(epoch, int) else str(epoch)

    if looks_classification:
        # Classification plotting
        y_true = targs.astype(int)
        y_pred = preds.astype(int)

        classes = np.unique(np.concatenate([y_true, y_pred], axis=0))
        classes = np.sort(classes)
        k = len(classes)

        # map labels to 0..k-1 for matrix
        idx = {c: i for i, c in enumerate(classes)}
        cm = np.zeros((k, k), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            if yt in idx and yp in idx:
                cm[idx[yt], idx[yp]] += 1

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Confusion matrix
        im = axes[0].imshow(cm, interpolation="nearest")
        axes[0].set_title(f"Confusion Matrix (Epoch {epoch})")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].set_xticks(range(k))
        axes[0].set_yticks(range(k))
        axes[0].set_xticklabels(classes)
        axes[0].set_yticklabels(classes)

        # annotate counts (small text; ok for k<=50)
        for i in range(k):
            for j in range(k):
                if cm[i, j] != 0:
                    axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

        # Per-class predicted counts
        pred_counts = np.array([(y_pred == c).sum() for c in classes])
        true_counts = np.array([(y_true == c).sum() for c in classes])

        x = np.arange(k)
        axes[1].bar(x - 0.2, true_counts, width=0.4, label="True")
        axes[1].bar(x + 0.2, pred_counts, width=0.4, label="Pred")
        axes[1].set_title(f"Per-class Counts (Epoch {epoch})")
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Count")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f"predictions_{epoch_str}.png", dpi=300, bbox_inches="tight")
        plt.close()
        return

    # Regression plotting (original behavior)
    predictions = preds.astype(np.float32)
    targets = targs.astype(np.float32)

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

