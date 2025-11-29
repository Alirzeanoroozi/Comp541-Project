# loaders/protein_dataset.py
import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        if self.labels is None:
            return {"protein": seq}
        else:
            return {"protein": seq, "label": torch.tensor(self.labels[idx])}
