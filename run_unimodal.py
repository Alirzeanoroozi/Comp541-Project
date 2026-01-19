import torch
import argparse
import os

from data.dataloaders import get_loaders
from utils.load_config import load_config
from utils.calculate_embeddings import calculate_embeddings
from models.unimodel import build_model
from trainer import RegressionTrainer, ClassificationTrainer


def main(name, dataset, max_len):
    config = load_config(f"{name}.yml")
    config["Dataset"] = dataset
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Task selection based on dataset (ONLY ecoli_proteins uses classification)
    is_classification = (config["Dataset"] == "ecoli_proteins")
    task = "classification" if is_classification else "regression"

    max_len = int(max_len)

    filtered_csv = f"data/datasets/{config['Dataset']}_multimodal_filtered_maxlen{max_len}.csv"
    emb_dir = f"embeddings/{config['Dataset']}/{config['modality']}/maxlen{max_len}"

    need_embeddings = not os.path.exists(os.path.join(emb_dir, "seq1.pt"))
    need_filtered_csv = not os.path.exists(filtered_csv)

    if need_embeddings or need_filtered_csv:
        print("Embeddings and/or filtered CSV not found, calculating...")
        calculate_embeddings(
            dataset=config["Dataset"],
            modality=config["modality"],
            device=config["device"],
            max_len=max_len,
        )

    print("=" * 60)
    print("Training Configuration:")
    print(f"  Name: {config['name']}")
    print(f"  Dataset: {config['Dataset']}")
    print(f"  Task: {task}")
    print(f"  Modality: {config['modality']}")
    print(f"  Max Len (filter): {max_len}")
    print("=" * 60)

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_loaders(
        config["Dataset"],
        32,
        modality=config["modality"],
        max_len=max_len,
    )

    print("\nInitializing model...")
    model = build_model(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    print(f"Total number of parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    print("\nInitializing trainer...")
    save_dir = f"./plots/{config['name']}/{config['Dataset']}"

    if is_classification:
        num_classes = int(config.get("num_classes", config.get("n_classes", 3)))
        trainer = ClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=config["device"],
            save_dir=save_dir,
            num_classes=num_classes,
            lr=float(config.get("lr", 3e-5)),
            weight_decay=float(config.get("weight_decay", 1e-5)),
        )
    else:
        trainer = RegressionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=config["device"],
            save_dir=save_dir,
        )

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
        help="Name of config file (without .yml extension) to use. Default: uni_rna",
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
