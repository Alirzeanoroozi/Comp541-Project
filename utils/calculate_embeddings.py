#!/usr/bin/env python3
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm

from models.rna_model import RNAFMEmbedder
from models.protein_model import ESM2Embedder
from models.dna_model import NucleotideTransformerEmbedder


def get_max_len(dataset, modality):
    df = pd.read_csv(f"data/datasets/{dataset}_multimodal.csv")
    seqs = df[modality].tolist()
    return max(len(str(seq)) for seq in seqs)

def save_embedding(embedding, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embedding.cpu(), save_path)

def ensure_filtered_csv(dataset, max_len):
    """
    Ensure a filtered CSV exists, uniquely keyed by max_len:
      data/datasets/{dataset}_multimodal_filtered_maxlen{max_len}.csv

    Filtering is always based on RNA length <= max_len.
    Reindexes 'id' as seq1, seq2, ... in the kept order.
    """
    csv_path = f"data/datasets/{dataset}_multimodal.csv"
    max_len = int(max_len)

    filtered_csv_path = csv_path.replace(".csv", f"_filtered_maxlen{max_len}.csv")

    if os.path.exists(filtered_csv_path):
        print(f"Filtered CSV already exists, using: {filtered_csv_path}")
        return filtered_csv_path

    df = pd.read_csv(csv_path)

    if "RNA" not in df.columns:
        raise KeyError(f"Column 'RNA' not found in {csv_path}. Available: {list(df.columns)}")
    if "id" not in df.columns:
        raise KeyError(f"Column 'id' not found in {csv_path}. Available: {list(df.columns)}")

    # filtering done based on RNA seq length
    rna_lengths = df["RNA"].astype(str).str.len()
    keep_mask = rna_lengths <= max_len

    kept_df = df.loc[keep_mask].copy().reset_index(drop=True)
    kept_df["id"] = [f"seq{i+1}" for i in range(len(kept_df))]

    kept_df.to_csv(filtered_csv_path, index=False)

    print("=" * 60)
    print("Created filtered CSV (RNA-length based).")
    print(f"Original rows : {len(df)}")
    print(f"Kept rows     : {len(kept_df)} (RNA length <= {max_len})")
    print(f"Saved to      : {filtered_csv_path}")
    print("=" * 60)

    return filtered_csv_path


def process_dataset_embedding(dataset, modality, model, filtered_csv_path, max_len):
    df = pd.read_csv(filtered_csv_path)

    if modality not in df.columns:
        raise KeyError(f"Column '{modality}' not found in {filtered_csv_path}. Available: {list(df.columns)}")
    if "id" not in df.columns:
        raise KeyError(f"Column 'id' not found in {filtered_csv_path}. Available: {list(df.columns)}")

    # keep embeddings separate per max_len to avoid mismatched files
    out_folder = os.path.join("embeddings", dataset, modality, f"maxlen{int(max_len)}")
    os.makedirs(out_folder, exist_ok=True)

    printed_shape = False

    for row in tqdm(df.itertuples(index=False), desc=f"Embedding {dataset} from {modality}", total=len(df)):
        seq = str(getattr(row, modality))
        _id = str(getattr(row, "id"))

        emb = model(seq)

        if not printed_shape:
            print(f"First embedding shape: {emb.shape}")
            printed_shape = True

        save_fp = os.path.join(out_folder, f"{_id}.pt")
        save_embedding(emb, save_fp)

    print(f"\nDone. Embedded {len(df)} sequences for modality={modality}.")
    print(f"Embeddings folder: {out_folder}")

def calculate_embeddings(dataset, modality, device, max_len=1024):
    print("Calculating embeddings... for dataset:", dataset, "and modality:", modality)
    print("=" * 60)

    max_len = int(max_len)
    print("Filtering rule for CSV: RNA length <=", max_len)
    filtered_csv_path = ensure_filtered_csv(dataset, max_len=max_len)

    # no max_len argument for embedders anymore, filtering is done with the csv
    if modality == "RNA":
        embedder = RNAFMEmbedder(device=device)
    elif modality == "Protein":
        embedder = ESM2Embedder(device=device)
    elif modality == "DNA":
        embedder = NucleotideTransformerEmbedder(device=device)
    else:
        raise ValueError(f"Invalid modality: {modality}")

    process_dataset_embedding(dataset, modality, embedder, filtered_csv_path=filtered_csv_path, max_len=max_len)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save embeddings. Creates filtered CSV keyed by max_len using RNA-length filter."
    )
    parser.add_argument("dataset", type=str, help="Dataset name (expects data/datasets/{dataset}_multimodal.csv)")
    parser.add_argument(
        "modality",
        type=str,
        choices=["RNA", "Protein", "DNA"],
        help="Which modality column to embed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run on (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="RNA-length filter threshold used to create *_filtered_maxlen{max_len}.csv (default: 1024)",
    )

    args = parser.parse_args()
    calculate_embeddings(dataset=args.dataset, modality=args.modality, device=args.device, max_len=args.max_len)


if __name__ == "__main__":
    main()
