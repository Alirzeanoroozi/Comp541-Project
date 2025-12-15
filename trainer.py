import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import time
from collections import deque


class RegressionTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, save_dir, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self.criterion = nn.MSELoss()

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Early stopping
        self.early_stopping_patience = 20
        self.early_stopping_counter = 0
        self.best_val_loss_for_early_stop = float('inf')

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.save_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mse': [], 'val_mse': [],
            'train_mae': [], 'val_mae': [],
            'train_r2': [], 'val_r2': [],
            'train_spearman': [], 'val_spearman': []
        }

        self.best_val_loss = float('inf')
        self.best_model_state = None

    # ------------------------------------------------------------------
    # INTERNAL BATCH HANDLER (NEW)
    # ------------------------------------------------------------------
    def _forward_batch(self, batch):
        if isinstance(batch, dict):
            targets = batch["label"].to(self.device).float()
            outputs = self.model(batch)
        else:
            inputs, targets = batch
            targets = targets.to(self.device).float()
            outputs = self.model(inputs)
        return outputs, targets

    # ------------------------------------------------------------------
    # TRAIN EPOCH
    # ------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        num_batches = len(self.train_loader)
        ema_batch_time = None
        alpha = 0.1
        last_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            now = time.time()
            batch_time = now - last_time
            last_time = now
            ema_batch_time = batch_time if ema_batch_time is None else alpha * batch_time + (1 - alpha) * ema_batch_time

            remaining_batches = num_batches - batch_idx
            eta_seconds = remaining_batches * ema_batch_time
            print(
                f"\rEpoch {epoch}: Training batch {batch_idx}/{num_batches}"
                f" — ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s",
                end=""
            )

            self.optimizer.zero_grad()
            outputs, targets = self._forward_batch(batch)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

        print()

        return self._compute_metrics(all_targets, all_preds, total_loss / num_batches)

    # ------------------------------------------------------------------
    # VALIDATE / TEST
    # ------------------------------------------------------------------
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in loader:
                outputs, targets = self._forward_batch(batch)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        return self._compute_metrics(all_targets, all_preds, total_loss / len(loader), return_preds=True)

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------
    def _compute_metrics(self, targets, preds, avg_loss, return_preds=False):
        targets = np.array(targets)
        preds = np.array(preds)

        mse = mean_squared_error(targets, preds)
        mae = mean_absolute_error(targets, preds)
        r2 = r2_score(targets, preds)

        try:
            spearman_corr, _ = spearmanr(targets, preds)
        except Exception:
            spearman_corr = 0.0

        out = {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'spearman': spearman_corr
        }

        if return_preds:
            out['predictions'] = preds
            out['targets'] = targets

        return out

    # ------------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------------
    def train(self, epochs):
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")

        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch + 1)
            val_metrics = self.validate(self.val_loader)

            self.scheduler.step(val_metrics['loss'])

            for k in train_metrics:
                self.history[f"train_{k}"].append(train_metrics[k])
                self.history[f"val_{k}"].append(val_metrics[k])

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, is_best=True)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train | Loss {train_metrics['loss']:.4f} | R² {train_metrics['r2']:.4f}")
            print(f"Val   | Loss {val_metrics['loss']:.4f} | R² {val_metrics['r2']:.4f}")


        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")

        # ✅ ADD THESE
        self.plot_training_curves()

        final_val = self.validate(self.val_loader)
        self.plot_predictions(
            final_val["predictions"],
            final_val["targets"],
            epoch="final"
        )


    # ------------------------------------------------------------------
    # CHECKPOINTING
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch, is_best):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')

        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
            print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")
            
            
    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def plot_training_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(epochs, self.history['train_mse'], label='Train')
        axes[0, 1].plot(epochs, self.history['val_mse'], label='Val')
        axes[0, 1].set_title('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(epochs, self.history['train_mae'], label='Train')
        axes[1, 0].plot(epochs, self.history['val_mae'], label='Val')
        axes[1, 0].set_title('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(epochs, self.history['train_r2'], label='Train')
        axes[1, 1].plot(epochs, self.history['val_r2'], label='Val')
        axes[1, 1].set_title('R²')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "training_curves.png", dpi=300)
        plt.close()

    def plot_predictions(self, predictions, targets, epoch):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(targets, predictions, alpha=0.5)
        min_v = min(targets.min(), predictions.min())
        max_v = max(targets.max(), predictions.max())
        axes[0].plot([min_v, max_v], [min_v, max_v], 'r--')
        axes[0].set_title("Predicted vs Actual")
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")

        residuals = predictions - targets
        axes[1].scatter(targets, residuals, alpha=0.5)
        axes[1].axhline(0, color='r', linestyle='--')
        axes[1].set_title("Residuals")

        plt.tight_layout()
        plt.savefig(self.plots_dir / f"predictions_{epoch}.png", dpi=300)
        plt.close()

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.history = ckpt.get('history', self.history)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            
