import torch
from torch.utils.data import Dataset
import os

class UnimodalDataset(Dataset):
    def __init__(self, embedding_folder, labels, ids):
        self.embedding_folder = embedding_folder
        self.labels = labels
        self.ids = ids
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        embedding = torch.load(os.path.join(self.embedding_folder, f"{self.ids[idx]}.pt"))
        return embedding, torch.tensor(self.labels[idx])
