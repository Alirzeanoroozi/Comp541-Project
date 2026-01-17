import torch
import argparse
import os
import pandas as pd
import shutil

from data.dataloaders import get_loaders
from data.subsampler.subsample import subsample_loader
from utils.load_config import load_config
from utils.calculate_embeddings import calculate_embeddings
from models.unimodel import build_model
from trainer import RegressionTrainer, ClassificationTrainer


def main(name, dataset, max_len):
    config = load_config(f"{name}.yml")
    config["task"] = "classification"
    config["num_classes"] = 3
    config["Dataset"] = dataset
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    max_len = int(max_len)

    filtered_csv = f"data/datasets/{config['Dataset']}_multimodal_filtered_maxlen{max_len}.csv"
    emb_dir = f"embeddings/{config['Dataset']}/{config['modality']}/maxlen{max_len}"

    need_embeddings = not os.path.exists(os.path.join(emb_dir, "seq1.pt"))
    need_filtered_csv = not os.path.exists(filtered_csv)

    # Paths
    filtered_csv = f"data/datasets/{config['Dataset']}_multimodal_filtered_maxlen{max_len}.csv"
    emb_dir = f"embeddings/{config['Dataset']}/{config['modality']}/maxlen{max_len}"

    need_filtered_csv = not os.path.exists(filtered_csv)
    need_embeddings = not os.path.exists(emb_dir)

    # 🔥 If CSV needs to be regenerated, embeddings are INVALID
    if need_filtered_csv and os.path.exists(emb_dir):
        print("Filtered CSV missing → removing stale embeddings")
        shutil.rmtree(emb_dir)

    # 🔥 If embeddings exist but CSV was recreated earlier → force rebuild
    if not need_filtered_csv and os.path.exists(emb_dir):
        df = pd.read_csv(filtered_csv)
        expected_ids = set(df["id"].astype(str))
        existing_ids = {
            f.replace(".pt", "") for f in os.listdir(emb_dir)
            if f.endswith(".pt")
        }

        if not expected_ids.issubset(existing_ids):
            print("Embedding mismatch detected → removing stale embeddings")
            shutil.rmtree(emb_dir)
            need_embeddings = True

    if need_embeddings or need_filtered_csv:
        print("Recomputing embeddings...")
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
    
    print(
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset)
    )

    train_loader = subsample_loader(train_loader, fraction=1)
    val_loader   = subsample_loader(val_loader, fraction=1)
    test_loader  = subsample_loader(test_loader, fraction=1)
    
    print(
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset)
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
    trainer = ClassificationTrainer(
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
        default="ecoli_proteins",
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
    args, _ = parser.parse_known_args()
    main(args.name, args.dataset, args.max_len)
