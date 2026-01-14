import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from tqdm import tqdm
import pandas as pd
import json

class RegressionTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
    
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        # Early stopping
        self.early_stopping_patience = 20
        self.early_stopping_counter = 0
        self.best_val_loss_for_early_stop = float('inf')
        
        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_spearman': [],
            'val_spearman': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n = 0
        all_preds = []
        all_targets = []

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()

            seqs, embeddings, labels, lengths, mask = batch

            embeddings = embeddings.to(self.device)          # [B, L, D]
            targets = labels.to(self.device).float().view(-1)  # [B]
            mask = mask.to(self.device)                      # [B, L] bool

            outputs = self.model(seqs, embeddings, mask=mask).float().view(-1)  # [B]

            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()
            
            # fixes possible skew due to last batch being smaller
            bs = targets.size(0)
            total_loss += loss.item() * bs
            n += bs

            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        mse = mean_squared_error(all_targets, all_preds)
        spearman_corr = spearmanr(all_targets, all_preds)[0]
        avg_loss = total_loss / max(n, 1)

        return {'loss': avg_loss, 'mse': mse, 'spearman': spearman_corr, 'lr': self.optimizer.param_groups[0]['lr']}

    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        n = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                seqs, embeddings, labels, lengths, mask = batch

                embeddings = embeddings.to(self.device)
                targets = labels.to(self.device).float().view(-1)
                mask = mask.to(self.device)

                outputs = self.model(seqs, embeddings, mask=mask).float().view(-1)

                loss = self.criterion(outputs, targets)

                bs = targets.size(0)
                total_loss += loss.item() * bs
                n += bs

                all_preds.extend(outputs.detach().cpu().tolist())
                all_targets.extend(targets.detach().cpu().tolist())

        all_preds = np.asarray(all_preds, dtype=np.float32)
        all_targets = np.asarray(all_targets, dtype=np.float32)

        mse = mean_squared_error(all_targets, all_preds)

        rho = spearmanr(all_targets, all_preds).correlation
        if not np.isfinite(rho):
            rho = 0.0

        avg_loss = total_loss / max(n, 1)

        return {
            "loss": avg_loss,
            "mse": mse,
            "spearman": rho,
            "predictions": all_preds,
            "targets": all_targets,
        }

    
    def train(self, epochs):
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate(self.val_loader)
            
            # Update scheduler
            #self.scheduler.step(val_metrics['loss'])
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['train_spearman'].append(train_metrics.get('spearman', 0.0))
            self.history['val_spearman'].append(val_metrics.get('spearman', 0.0))
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"Spearman: {train_metrics.get('spearman', 0.0):.4f}, "
                  f"LR: {train_metrics['lr']:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"MSE: {val_metrics['mse']:.4f}, "
                  f"Spearman: {val_metrics.get('spearman', 0.0):.4f}")
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss_for_early_stop:
                self.best_val_loss_for_early_stop = val_metrics['loss']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                    print(f"Best validation loss: {self.best_val_loss_for_early_stop:.4f}")
                    break
            
            # Plot progress every 5 epochs or on last epoch
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.plot_training_curves()
                self.plot_predictions(val_metrics['predictions'], val_metrics['targets'], epoch=epoch+1)
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")
        
        # Final plots
        self.plot_training_curves()
        final_val_metrics = self.validate(self.val_loader)
        self.plot_predictions(final_val_metrics['predictions'], final_val_metrics['targets'], epoch='final')
        
        test_metrics = self.validate(self.test_loader)
        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        print(f"  Spearman: {test_metrics.get('spearman', 0.0):.4f}")
        self.plot_predictions(test_metrics['predictions'], test_metrics['targets'], epoch='test')
        self.save_values(final_val_metrics, epoch='final')
        self.save_values(test_metrics, epoch='test')
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MSE curves
        axes[1].plot(epochs, self.history['train_mse'], 'b-', label='Train MSE')
        axes[1].plot(epochs, self.history['val_mse'], 'r-', label='Val MSE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].set_title('Training and Validation MSE')
        axes[1].legend()
        axes[1].grid(True)
        
        # Spearman curves
        axes[2].plot(epochs, self.history['train_spearman'], 'b-', label='Train Spearman')
        axes[2].plot(epochs, self.history['val_spearman'], 'r-', label='Val Spearman')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Spearman')
        axes[2].set_title('Training and Validation Spearman')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions(self, predictions, targets, epoch):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].scatter(targets, predictions, alpha=0.5, s=10)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'Predictions vs Actual (Epoch {epoch})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        residuals = predictions - targets
        axes[1].scatter(targets, residuals, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Residuals (Predicted - Actual)')
        axes[1].set_title(f'Residual Plot (Epoch {epoch})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        epoch_str = f'epoch_{epoch}' if epoch else 'final'
        plt.savefig(self.save_dir / f'predictions_{epoch_str}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # helper for save_values
    def _best_min(self, values):
        """Return (best_value, best_epoch_1based) for a metric where lower is better."""
        if not values:
            return None, None
        arr = np.asarray(values, dtype=np.float64)
        idx = int(np.nanargmin(arr))
        return float(arr[idx]), idx + 1

    # helper for save_values
    def _best_max(self, values):
        """Return (best_value, best_epoch_1based) for a metric where higher is better."""
        if not values:
            return None, None
        arr = np.asarray(values, dtype=np.float64)
        idx = int(np.nanargmax(arr))
        return float(arr[idx]), idx + 1

    def save_values(self, metrics, epoch):
        predictions = metrics.get("predictions")
        targets = metrics.get("targets")
        if predictions is not None and targets is not None:
            df = pd.DataFrame({
                "predictions": predictions,
                "targets": targets
            })
            df.to_csv(self.save_dir / f"{epoch}_predictions_vs_targets.csv", index=False)
        
        # record best metrics
        best_train_loss, best_train_loss_ep = self._best_min(self.history.get("train_loss", []))
        best_val_loss, best_val_loss_ep = self._best_min(self.history.get("val_loss", []))
        best_train_mse, best_train_mse_ep = self._best_min(self.history.get("train_mse", []))
        best_val_mse, best_val_mse_ep = self._best_min(self.history.get("val_mse", []))
        best_train_spearman, best_train_spearman_ep = self._best_max(self.history.get("train_spearman", []))
        best_val_spearman, best_val_spearman_ep = self._best_max(self.history.get("val_spearman", []))
        summary_path = self.save_dir / "metrics_summary.json"

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

        # Also record the metrics from this call (e.g., "final", "test")
        run_metrics = {k: v for k, v in metrics.items() if k not in ("predictions", "targets")}
        summary.setdefault("runs", {})
        summary["runs"][str(epoch)] = run_metrics

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
