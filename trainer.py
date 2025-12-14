import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

class RegressionTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, save_dir, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
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
        
        # Entropy regularization weight
        self.entropy_lambda = 0.0
        
        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.save_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_mae': [],
            'val_mae': [],
            'train_r2': [],
            'val_r2': [],
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
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            seqs, embeddings, targets = batch
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device).float()
            outputs = self.model(embeddings)
            
            mse_loss = self.criterion(outputs, targets)
            
            entropy_loss = 0.0
            if self.entropy_lambda > 0:
                # Try to get entropy from model if it's a fusion model with attention
                if hasattr(self.model, 'get_entropy'):
                    entropy_loss = self.model.get_entropy()
                # For MIL fusion, compute entropy from attention weights
                elif hasattr(self.model, 'fusion') and hasattr(self.model.fusion, 'attn'):
                    # This is a simplified entropy computation
                    # In practice, you'd need to extract attention weights during forward pass
                    pass
            
            loss = mse_loss + self.entropy_lambda * entropy_loss
            
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
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        # Compute Spearman correlation
        try:
            spearman_corr, _ = spearmanr(all_targets, all_preds)
        except:
            spearman_corr = 0.0
        
        avg_loss = total_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'spearman': spearman_corr
        }
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                seqs, embeddings, targets = batch
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device).float()
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        # Compute Spearman correlation
        try:
            print(all_targets)
            print(all_preds)
            spearman_corr, _ = spearmanr(all_targets, all_preds)
        except:
            print("Error in Spearman correlation")
            spearman_corr = 0.0
        
        avg_loss = total_loss / len(loader)
        
        return {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'spearman': spearman_corr,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def train(self, epochs):
        num_epochs = epochs or self.config.get('epochs', 30)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(num_epochs):
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
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['train_r2'].append(train_metrics['r2'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['train_spearman'].append(train_metrics.get('spearman', 0.0))
            self.history['val_spearman'].append(val_metrics.get('spearman', 0.0))
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, is_best=True)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"MAE: {train_metrics['mae']:.4f}, "
                  f"R²: {train_metrics['r2']:.4f}, "
                  f"Spearman: {train_metrics.get('spearman', 0.0):.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"MSE: {val_metrics['mse']:.4f}, "
                  f"MAE: {val_metrics['mae']:.4f}, "
                  f"R²: {val_metrics['r2']:.4f}, "
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
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.plot_training_curves()
                self.plot_predictions(val_metrics['predictions'], 
                                     val_metrics['targets'], 
                                     epoch=epoch+1)
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")
        
        # Final plots
        self.plot_training_curves()
        final_val_metrics = self.validate(self.val_loader)
        self.plot_predictions(final_val_metrics['predictions'], 
                             final_val_metrics['targets'], 
                             epoch='final')
        
        # Test evaluation if available
        if self.test_loader:
            test_metrics = self.validate(self.test_loader)
            print(f"\nTest Results:")
            print(f"  Loss: {test_metrics['loss']:.4f}")
            print(f"  MSE: {test_metrics['mse']:.4f}")
            print(f"  MAE: {test_metrics['mae']:.4f}")
            print(f"  R²: {test_metrics['r2']:.4f}")
            print(f"  Spearman: {test_metrics.get('spearman', 0.0):.4f}")
            self.plot_predictions(test_metrics['predictions'], 
                                 test_metrics['targets'], 
                                 epoch='test')
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MSE curves
        axes[0, 1].plot(epochs, self.history['train_mse'], 'b-', label='Train MSE')
        axes[0, 1].plot(epochs, self.history['val_mse'], 'r-', label='Val MSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_title('Training and Validation MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MAE curves
        axes[1, 0].plot(epochs, self.history['train_mae'], 'b-', label='Train MAE')
        axes[1, 0].plot(epochs, self.history['val_mae'], 'r-', label='Val MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Training and Validation MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # R² curves
        axes[1, 1].plot(epochs, self.history['train_r2'], 'b-', label='Train R²')
        axes[1, 1].plot(epochs, self.history['val_r2'], 'r-', label='Val R²')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Training and Validation R²')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(self.plots_dir / f'predictions_{epoch_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch, is_best):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']

