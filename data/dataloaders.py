import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.loaders.unimodal_dataset import UnimodalDataset

def get_loaders(name, batch_size=32, modality="RNA", max_len=None):
    suffix = f"_maxlen{max_len}" if max_len is not None else ""
    df = pd.read_csv(f"data/datasets/{name}_multimodal_filtered{suffix}.csv")

    df["Value"] = df["Value"].astype(float)

    train_df = df[df["Split"] == "train"]
    val_df   = df[df["Split"] == "val"]
    test_df  = df[df["Split"] == "test"]

    embedding_folder = f"embeddings/{name}/{modality}"
    if max_len is not None:
        embedding_folder = f"{embedding_folder}/maxlen{max_len}"

    train_dataset = UnimodalDataset(embedding_folder, train_df["Value"].tolist(), train_df["id"].tolist())
    val_dataset   = UnimodalDataset(embedding_folder, val_df["Value"].tolist(),   val_df["id"].tolist())
    test_dataset  = UnimodalDataset(embedding_folder, test_df["Value"].tolist(),  test_df["id"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_unimodal)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_unimodal)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_unimodal)

    return train_loader, val_loader, test_loader

# since the sequences are of variable lengths, we need a custom collate function for padding
def collate_unimodal(batch):
    # batch items: (embedding([L,D]), label(tensor))
    embs, labels = zip(*batch)

    # ensure float tensors
    embs = [e.float() for e in embs]

    lengths = torch.tensor([e.size(0) for e in embs], dtype=torch.long)
    B = len(embs)
    Lmax = int(lengths.max().item())
    D = embs[0].size(1)

    padded = embs[0].new_zeros((B, Lmax, D))  # zeros padding
    for i, e in enumerate(embs):
        padded[i, : e.size(0), :] = e

    # mask: True for real tokens, False for padding
    mask = torch.arange(Lmax).unsqueeze(0) < lengths.unsqueeze(1)  # [B, Lmax], bool

    labels = torch.stack(labels).float()  # [B]
    return padded, labels, lengths, mask

