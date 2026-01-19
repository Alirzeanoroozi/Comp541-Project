import torch
from torch.utils.data import Dataset

class LoRADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = torch.tensor(self.labels[idx])
        return sequence, label