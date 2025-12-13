import torch
from torch.utils.data import Dataset
import os

class UnimodalDataset(Dataset):
    def __init__(self, embedding_folder, sequences, labels):
        self.embedding_folder = embedding_folder
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        try:
            embedding = torch.load(os.path.join(self.embedding_folder, f"seq{idx+1}.pt"))
        except FileNotFoundError:
            embedding = torch.zeros(350, 640)
        return seq, embedding, torch.tensor(self.labels[idx])
