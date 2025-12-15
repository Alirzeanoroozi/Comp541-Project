import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.loaders.unimodal_dataset import UnimodalDataset
from data.loaders.multimodal_dataset import MultiModalDataset


def multimodal_collate_fn(batch):
    return {
        "dna": [item["dna"] for item in batch],
        "rna": [item["rna"] for item in batch],
        "protein": [item["protein"] for item in batch],
        "label": torch.stack([item["label"] for item in batch])
    }


def get_loaders(name, batch_size, modality=None, subset_pct=1.0):

    # ---------------------------
    # LOAD CSV
    # ---------------------------
    if modality is not None and modality.lower() == "multi":
        csv_path = f"data/datasets/{name}_multimodal.csv"
    else:
        csv_path = f"data/datasets/{name}.csv"

    df = pd.read_csv(csv_path)

    if subset_pct < 1.0:
        df = df.sample(frac=subset_pct, random_state=42)

    df["Value"] = df["Value"].astype(float)

    train_df = df[df["Split"] == "train"]
    val_df   = df[df["Split"] == "val"]
    test_df  = df[df["Split"] == "test"]

    # ---------------------------
    # MULTIMODAL
    # ---------------------------
    if modality.lower() == "multi":

        train_dataset = MultiModalDataset(
            dna_seqs=train_df["DNA"].tolist(),
            rna_seqs=train_df["RNA"].tolist(),
            protein_seqs=train_df["Protein"].tolist(),
            labels=train_df["Value"].tolist()
        )

        val_dataset = MultiModalDataset(
            dna_seqs=val_df["DNA"].tolist(),
            rna_seqs=val_df["RNA"].tolist(),
            protein_seqs=val_df["Protein"].tolist(),
            labels=val_df["Value"].tolist()
        )

        test_dataset = MultiModalDataset(
            dna_seqs=test_df["DNA"].tolist(),
            rna_seqs=test_df["RNA"].tolist(),
            protein_seqs=test_df["Protein"].tolist(),
            labels=test_df["Value"].tolist()
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=multimodal_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=multimodal_collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=multimodal_collate_fn
        )

    # ---------------------------
    # UNIMODAL
    # ---------------------------
    else:
        if modality is None:
            raise ValueError("modality must be specified for unimodal training")

        train_dataset = UnimodalDataset(
            sequences=train_df["Sequence"].tolist(),
            labels=train_df["Value"].tolist()
        )

        val_dataset = UnimodalDataset(
            sequences=val_df["Sequence"].tolist(),
            labels=val_df["Value"].tolist()
        )

        test_dataset = UnimodalDataset(
            sequences=test_df["Sequence"].tolist(),
            labels=test_df["Value"].tolist()
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
