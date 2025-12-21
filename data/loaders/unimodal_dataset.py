import torch
from torch.utils.data import Dataset
import os

class UnimodalDataset(Dataset):
    def __init__(self, embedding_folder, sequences, labels, ids):
        self.embedding_folder = embedding_folder
        self.sequences = sequences
        self.labels = labels
        self.ids = ids
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        embedding = torch.load(os.path.join(self.embedding_folder, f"{self.ids[idx]}.pt"))
        return seq, embedding, torch.tensor(self.labels[idx])
