# loaders/dna_dataset.py
import torch
from torch.utils.data import Dataset

class DNADataset(Dataset):
    def __init__(self, sequences, labels=None):
        """
        sequences: list[str]
        labels: list[int] or list[float] or None
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        if self.labels is None:
            return {"dna": seq}
        else:
            return {"dna": seq, "label": torch.tensor(self.labels[idx])}
