import torch
import argparse

from data.dataloaders import get_loaders
from utils.load_config import load_config
from models.unimodel import build_model
from trainer import RegressionTrainer

def main(name):
    config = load_config(f'{name}.yml')
    if "uni" in name:
        if "rna" in name:
            config['modality'] = 'RNA'
            config['max_len'] = 1000
            config['embedding_dim'] = 640
        elif "protein" in name:
            config['modality'] = 'Protein'
            config['max_len'] = 1024
            config['embedding_dim'] = 320
        elif "dna" in name:
            config['modality'] = 'DNA'
            config['max_len'] = 512
            config['embedding_dim'] = 4107
    config['Dataset'] = 'fungal_expression' #  'mrna_stability', 'ecoli_proteins', 'cov_vaccine_degradation', 'fungal_expression'
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
        save_dir=f"./checkpoints/{config['name']}/{config['Dataset']}",
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train(epochs=500)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unimodal model")
    parser.add_argument(
        "--name",
        type=str,
        default="uni_rna",
        help="Name of config file (without .yml extension) to use. Default: uni_rna"
    )
    args = parser.parse_args()
    main(args.name)

