# loaders/rna_dataset.py
import torch
from torch.utils.data import Dataset

class UnimodalDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq, torch.tensor(self.labels[idx])
