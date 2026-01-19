import torch
import argparse
import os

from data.dataloaders import get_lora_loaders
from utils.load_config import load_config
from utils.calculate_embeddings import calculate_embeddings
from models.lora_model import build_model
from trainer import RegressionTrainer


def main(name, dataset, max_len):
    config = load_config(f"{name}.yml")
    config["Dataset"] = dataset
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    max_len = int(max_len)

    print("=" * 60)
    print("Training Configuration:")
    print(f"  Name: {config['name']}")
    print(f"  Dataset: {config['Dataset']}")
    print(f"  Modality: {config['modality']}")
    print(f"  Max Len (filter): {max_len}")
    print("=" * 60)

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_lora_loaders(config["Dataset"], 32, max_len=max_len, modality=config["modality"])

    print("\nInitializing model...")
    model = build_model(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    print(f"Total number of parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    print("\nInitializing trainer...")
    trainer = RegressionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config["device"],
        save_dir=f"./plots/{config['name']}/{config['Dataset']}",
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train(epochs=500)

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA model")
    parser.add_argument(
        "--name",
        type=str,
        default="lora_rna",
        help="Name of config file (without .yml extension) to use. Default: lora_rna",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fungal_expression",
        help=(
            "Dataset to use. Default: fungal_expression. Options: "
            "'mrna_stability', 'ecoli_proteins', 'cov_vaccine_degradation', 'fungal_expression'"
        ),
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Filter threshold: keep only sequences with raw length <= max_len before embedding/training.",
    )
    args = parser.parse_args()
    main(args.name, args.dataset, args.max_len)
