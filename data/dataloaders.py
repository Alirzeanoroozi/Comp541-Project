import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.loaders.unimodal_dataset import UnimodalDataset
from data.loaders.multimodal_dataset import MultiModalDataset
from data.loaders.lora_dataset import LoRADataset

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

    labels = torch.stack(labels).float().view(-1)  # [B]
    return padded, labels, lengths, mask


def get_multimodal_loaders(name, batch_size=32, max_len=None):
    suffix = f"_maxlen{max_len}" if max_len is not None else ""
    df = pd.read_csv(f"data/datasets/{name}_multimodal_filtered{suffix}.csv")
    df["Value"] = df["Value"].astype(float)

    train_df = df[df["Split"] == "train"]
    val_df   = df[df["Split"] == "val"]
    test_df  = df[df["Split"] == "test"]

    # folders
    dna_folder = f"embeddings/{name}/DNA"
    rna_folder = f"embeddings/{name}/RNA"
    prot_folder = f"embeddings/{name}/Protein"
    if max_len is not None:
        dna_folder  = f"{dna_folder}/maxlen{max_len}"
        rna_folder  = f"{rna_folder}/maxlen{max_len}"
        prot_folder = f"{prot_folder}/maxlen{max_len}"

    train_dataset = MultiModalDataset(
        dna_folder=dna_folder,
        rna_folder=rna_folder,
        protein_folder=prot_folder,
        labels=train_df["Value"].tolist(),
        ids=train_df["id"].tolist(),
    )
    val_dataset = MultiModalDataset(
        dna_folder=dna_folder,
        rna_folder=rna_folder,
        protein_folder=prot_folder,
        labels=val_df["Value"].tolist(),
        ids=val_df["id"].tolist(),
    )
    test_dataset = MultiModalDataset(
        dna_folder=dna_folder,
        rna_folder=rna_folder,
        protein_folder=prot_folder,
        labels=test_df["Value"].tolist(),
        ids=test_df["id"].tolist(),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_multimodal)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)

    return train_loader, val_loader, test_loader


def _pad_stack(embs):
    """
    embs: list of [L_i, D] float tensors (variable L, fixed D per modality)
    returns:
      padded:  [B, Lmax, D]
      lengths: [B]
      mask:    [B, Lmax] bool (True=real, False=pad)
    """
    embs = [e.float() for e in embs]

    lengths = torch.tensor([e.size(0) for e in embs], dtype=torch.long)
    B = len(embs)
    Lmax = int(lengths.max().item())
    D = embs[0].size(1)

    padded = embs[0].new_zeros((B, Lmax, D))
    for i, e in enumerate(embs):
        padded[i, : e.size(0), :] = e

    mask = torch.arange(Lmax, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, lengths, mask

def collate_multimodal(batch):
    dna_embs, rna_embs, prot_embs, labels = zip(*batch)

    dna_pad, dna_len, dna_mask = _pad_stack(dna_embs)
    rna_pad, rna_len, rna_mask = _pad_stack(rna_embs)
    prot_pad, prot_len, prot_mask = _pad_stack(prot_embs)

    labels = torch.stack(labels).float().view(-1)  # [B]

    return (
        dna_pad, rna_pad, prot_pad, labels,
        dna_len, rna_len, prot_len,
        dna_mask, rna_mask, prot_mask
    )

def get_lora_loaders(name, batch_size=32, max_len=None, modality="RNA"):
    suffix = f"_maxlen{max_len}" if max_len is not None else ""
    df = pd.read_csv(f"data/datasets/{name}_multimodal_filtered{suffix}.csv")
    df["Value"] = df["Value"].astype(float)

    train_df = df[df["Split"] == "train"]
    val_df   = df[df["Split"] == "val"]
    test_df  = df[df["Split"] == "test"]

    train_dataset = LoRADataset(train_df[modality].tolist(), train_df["Value"].tolist())
    val_dataset = LoRADataset(val_df[modality].tolist(), val_df["Value"].tolist())
    test_dataset = LoRADataset(test_df[modality].tolist(), test_df["Value"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader