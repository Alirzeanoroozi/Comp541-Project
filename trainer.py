import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from tqdm import tqdm

class RegressionTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
    
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
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
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            seqs, embeddings, targets = batch
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device).float()
            outputs = self.model(seqs, embeddings)
            
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mse = mean_squared_error(all_targets, all_preds)
        
        spearman_corr = spearmanr(all_targets, all_preds)[0]
        
        avg_loss = total_loss / len(self.train_loader)
        
        return {'loss': avg_loss, 'mse': mse, 'spearman': spearman_corr, 'lr': self.optimizer.param_groups[0]['lr']}
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                seqs, embeddings, targets = batch
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device).float()
                outputs = self.model(seqs, embeddings)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mse = mean_squared_error(all_targets, all_preds)
        
        spearman_corr = spearmanr(all_targets, all_preds)[0]
        
        avg_loss = total_loss / len(loader)
        
        return {'loss': avg_loss, 'mse': mse, 'spearman': spearman_corr, 'predictions': all_preds, 'targets': all_targets}
    
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
            self.scheduler.step(val_metrics['loss'])
            
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
