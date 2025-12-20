import torch
import argparse
import os

from data.dataloaders import get_loaders
from utils.load_config import load_config
from utils.calculate_embeddings import calculate_embeddings
from models.unimodel import build_model
from trainer import RegressionTrainer

def main(name, dataset):
    config = load_config(f'{name}.yml')
    config['Dataset'] = dataset
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # check if the embeddings are calculated
    if not os.path.exists(f"embeddings/{config['Dataset']}/{config['modality']}"):
        print("Embeddings not found, calculating...")
        calculate_embeddings(config['Dataset'], config['modality'])
    
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Name: {config['name']}")
    print(f"  Dataset: {config['Dataset']}")
    print(f"  Modality: {config['modality']}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_loaders(config['Dataset'], 32, modality=config['modality'])
    
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
        save_dir=f"./plots/{config['name']}/{config['Dataset']}",
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="fungal_expression",
        help="Dataset to use. Default: fungal_expression. Options: 'mrna_stability', 'ecoli_proteins', 'cov_vaccine_degradation', 'fungal_expression'"
    )
    args = parser.parse_args()
    main(args.name, args.dataset)

