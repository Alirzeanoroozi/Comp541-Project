import os
import torch
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    """
    Loads 3 modality embeddings from disk.

    Returns:
      dna_emb, rna_emb, prot_emb, label
    """
    def __init__(self, dna_folder, rna_folder, protein_folder, labels, ids):
        assert len(labels) == len(ids)

        self.dna_folder = dna_folder
        self.rna_folder = rna_folder
        self.protein_folder = protein_folder

        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]

        dna_emb  = torch.load(os.path.join(self.dna_folder, f"{_id}.pt")).float()
        rna_emb  = torch.load(os.path.join(self.rna_folder, f"{_id}.pt")).float()
        prot_emb = torch.load(os.path.join(self.protein_folder, f"{_id}.pt")).float()

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return dna_emb, rna_emb, prot_emb, label
