import torch
import sys

from data.dataloaders import get_loaders
from utils.load_config import load_config
from models.unimodel import build_model
from trainer import RegressionTrainer

def main(name):
    config = load_config(f'{name}.yml')

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Name: {config['name']}")
    print(f"  Dataset: {config['Dataset']}")
    print(f"  Modality: {config['modality']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_loaders(config['Dataset'], config['batch_size'], modality=config['modality'])
    
    # Initialize model
    print("\nInitializing model...")
    model = build_model(config)
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = RegressionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config['device'],
        save_dir=f"./checkpoints/{config['name']}/{config['modality']}",
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
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = 'uni_rna'
    main(name)

