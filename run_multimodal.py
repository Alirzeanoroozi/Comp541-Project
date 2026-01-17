import os
import argparse
import torch

from data.dataloaders import get_multimodal_loaders
from utils.load_config import load_config
from utils.calculate_embeddings import calculate_embeddings
from models.multimodel import build_model
from trainer import RegressionTrainer


def _has_any_embeddings(emb_dir) -> bool:
    if not os.path.isdir(emb_dir):
        return False
    try:
        for fn in os.listdir(emb_dir):
            if fn.endswith(".pt"):
                return True
    except FileNotFoundError:
        return False
    return False


def main(name: str, dataset: str, max_len: int, batch_size: int, epochs: int):
    config = load_config(f"{name}.yml")
    config["Dataset"] = dataset
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    max_len = int(max_len)

    filtered_csv = f"data/datasets/{config['Dataset']}_multimodal_filtered_maxlen{max_len}.csv"

    dna_dir = f"embeddings/{config['Dataset']}/DNA/maxlen{max_len}"
    rna_dir = f"embeddings/{config['Dataset']}/RNA/maxlen{max_len}"
    prot_dir = f"embeddings/{config['Dataset']}/Protein/maxlen{max_len}"

    need_filtered_csv = not os.path.exists(filtered_csv)
    need_dna = not _has_any_embeddings(dna_dir)
    need_rna = not _has_any_embeddings(rna_dir)
    need_prot = not _has_any_embeddings(prot_dir)
    need_embeddings = need_dna or need_rna or need_prot

    if need_filtered_csv or need_embeddings:
        print("Embeddings and/or filtered CSV not found, calculating...")
        for modality in ("DNA", "RNA", "Protein"):
            calculate_embeddings(
                dataset=config["Dataset"],
                modality=modality,
                device=config["device"],
                max_len=max_len,
            )

    print("=" * 60)
    print("Training Configuration (Multimodal):")
    print(f"  Config name: {config.get('name', name)}")
    print(f"  Dataset: {config['Dataset']}")
    print(f"  Max Len (filter): {max_len}")
    print(f"  Fusion: {config.get('fusion_type', 'concat')}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {config['device']}")
    print("=" * 60)

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_multimodal_loaders(
        config["Dataset"],
        batch_size=batch_size,
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
    trainer = RegressionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config["device"],
        save_dir=f"./plots/{config.get('name', name)}/{config['Dataset']}",
    )

    # entropy reg for MIL (nested config with backwards-compatible fallback)
    lam_entropy = None
    if isinstance(config.get("trainer", None), dict):
        lam_entropy = config["trainer"].get("lam_entropy", None)
    if lam_entropy is None:
        lam_entropy = config.get("lam_entropy", None)

    if lam_entropy is not None:
        trainer.lam_entropy = float(lam_entropy)
        if trainer.lam_entropy > 0:
            print(f"Using MIL entropy regularization: lam_entropy={trainer.lam_entropy}")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train(epochs=epochs)

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal model (DNA+RNA+Protein)")
    parser.add_argument(
        "--name",
        type=str,
        default="fusion_concat",
        help="Config file name (without .yml). Options: fusion_concat, fusion_mil, fusion_xattn",
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
        default=1000,
        help="Filter threshold: keep only sequences with raw length <= max_len before embedding/training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs.",
    )
    args = parser.parse_args()

    main(
        name=args.name,
        dataset=args.dataset,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

