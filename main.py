import torch
import torch.nn as nn
import torch.optim as optim
from trainer import RegressionTrainer
from models.rna_model import RNAFMEmbedder

import sys
import os

from data.dataloaders import get_loaders

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix path for dataloaders
import pandas as pd
from torch.utils.data import DataLoader
from data.loaders.rna_dataset import RNADataset


class BatchMLPHead(nn.Module):
    """Batch-compatible MLP head for regression."""
    def __init__(self, input_dim=256, num_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (B, D) - already batched and pooled
        return self.net(x).squeeze(-1)  # (B,)


class RNARegressionModel(nn.Module):
    """Simple RNA regression model combining embedder and prediction head."""
    
    def __init__(self, embedder, prediction_head):
        super().__init__()
        self.embedder = embedder
        self.prediction_head = prediction_head
    
    def forward(self, rna_seqs):
        # Handle both single string and list of strings
        if isinstance(rna_seqs, str):
            rna_seqs = [rna_seqs]
        elif isinstance(rna_seqs, list):
            pass  # Already a list
        else:
            # Assume it's a batch from DataLoader (list of strings)
            rna_seqs = list(rna_seqs) if not isinstance(rna_seqs, list) else rna_seqs
        
        # Process each sequence and stack embeddings
        batch_embeddings = []
        for seq in rna_seqs:
            embeddings = self.embedder(seq)  # (L, D)
            # Average pool over sequence length to get fixed-size representation
            pooled = embeddings.mean(dim=0)  # (D,)
            batch_embeddings.append(pooled)
        
        # Stack to form batch
        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # (B, D)
        
        # Get predictions
        predictions = self.prediction_head(batch_embeddings)  # (B,)
        return predictions


def main():
    # Configuration based on the training setup requirements
    config = {
        'learning_rate': 3e-5,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'epochs': 500,
        'scheduler': 'reducelronplateau',
        'scheduler_patience': 5,
        'early_stopping_patience': 20,
        'entropy_lambda': 0.0,  # Set to > 0 if using entropy regularization
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'loss': 'mse'
    }
    
    # Dataset name - change this to use different datasets
    # Available: 'mrna_stability', 'ecoli_proteins', 'cov_vaccine_degradation', 'fungal_expression'
    dataset_name = 'fungal_expression'
    
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Weight Decay: {config['weight_decay']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Max Epochs: {config['epochs']}")
    print(f"  Scheduler: {config['scheduler']} (patience={config['scheduler_patience']})")
    print(f"  Early Stopping: patience={config['early_stopping_patience']}")
    print(f"  Device: {config['device']}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_loaders(dataset_name, config['batch_size'])
    
    # Initialize model
    print("\nInitializing model...")
    device = config['device']
    
    # RNA embedder (frozen)
    embedder = RNAFMEmbedder(max_len=512, device=device)
    
    # Get embedding dimension from the model
    # ESM2_t6_8M_UR50D has 320 dimensions (verified from notebook)
    embed_dim = 320
    
    # Prediction head (batch-compatible version)
    prediction_head = BatchMLPHead(input_dim=embed_dim, num_classes=1)
    
    # Combined model
    model = RNARegressionModel(embedder, prediction_head).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Setup loss function
    criterion = nn.MSELoss()
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = RegressionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=f"./checkpoints/{dataset_name}",
        config=config
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train(epochs=config['epochs'])
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

