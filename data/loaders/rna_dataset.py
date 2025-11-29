# loaders/rna_dataset.py
import torch
from torch.utils.data import Dataset

class RNADataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        if self.labels is None:
            return {"rna": seq}
        else:
            return {"rna": seq, "label": torch.tensor(self.labels[idx])}
