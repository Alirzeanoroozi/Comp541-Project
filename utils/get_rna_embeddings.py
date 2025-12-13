from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch

from models.rna_model import RNAFMEmbedder


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def embed_rna_sequences(
    embedder: RNAFMEmbedder,
    seqs: List[str],
    batch_size: int,
) -> torch.Tensor:
    """
    RNAFMEmbedder.forward(seq) -> (L, D)
    Returns stacked token embeddings: (N, L, D).
    """
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        for s in batch:
            emb = embedder(s)  # (L, D)
            out.append(emb.detach().cpu().unsqueeze(0))  # (1, L, D)
    return torch.cat(out, dim=0)  # (N, L, D)


def main():
    ap = argparse.ArgumentParser(description="Precompute RNA-FM embeddings from a single CSV into one .pt file.")
    ap.add_argument("--csv", type=str, required=True, help="Path to input CSV (must contain 'RNA' column).")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs.")

    ap.add_argument("--split_col", type=str, default="Split",
                    help="Split column name (e.g., train/val/test). If missing, all rows are labeled 'all'.")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Load data
    df = pd.read_csv(csv_path)
    if "RNA" not in df.columns:
        raise ValueError(f"'RNA' column not found in {csv_path}. Found columns: {list(df.columns)}")

    # Ensure split column exists (or create it)
    if args.split_col in df.columns:
        split_vals = df[args.split_col].fillna("all").astype(str).tolist()
    else:
        split_vals = ["all"] * len(df)

    # Stable row id for later joining
    df = df.reset_index(drop=False).rename(columns={"index": "_row_id"})

    # Output folder per CSV stem
    base_out = out_dir / csv_path.stem
    ensure_dir(base_out)

    # Save an index file with all original columns (+ _row_id)
    index_path = base_out / f"{csv_path.stem}_index.csv"
    df.to_csv(index_path, index=False)
    print(f"Saved index: {index_path}")

    # Initialize embedder
    embedder = RNAFMEmbedder(max_len=args.max_len, device=args.device)
    embedder.eval()

    # Embed all rows in original order
    seqs = df["RNA"].fillna("").astype(str).tolist()
    emb = embed_rna_sequences(embedder, seqs, batch_size=args.batch_size)  # (N, L, D)

    # Single .pt output
    out_path = base_out / "RNA_embeddings_tokens.pt"
    payload = {
        "row_id": df["_row_id"].to_numpy(),
        "split": split_vals,   # list[str], length N
        "emb": emb,            # (N, L, D)
        "max_len": args.max_len,
        "embedding_dim": emb.shape[-1],
    }
    torch.save(payload, out_path)

    print(f"Saved: {out_path}  shape={tuple(emb.shape)}")
    print("Done.")


if __name__ == "__main__":
    main()
