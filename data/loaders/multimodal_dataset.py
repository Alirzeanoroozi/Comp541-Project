# loaders/multimodal_dataset.py
import torch
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    """
    Each item returns:
    {
        "dna": DNA sequence (str),
        "rna": RNA sequence (str),
        "protein": AA sequence (str),
        "text": optional text description (str)
        "label": float or int
    }
    """

    def __init__(self, dna_seqs, rna_seqs, protein_seqs,
                 text_seqs=None, labels=None):
        
        assert len(dna_seqs) == len(rna_seqs) == len(protein_seqs)

        if text_seqs is not None:
            assert len(text_seqs) == len(dna_seqs)

        if labels is not None:
            assert len(labels) == len(dna_seqs)

        self.dna = dna_seqs
        self.rna = rna_seqs
        self.protein = protein_seqs
        self.text = text_seqs
        self.labels = labels

    def __len__(self):
        return len(self.dna)

    def __getitem__(self, idx):
        item = {
            "dna": self.dna[idx],
            "rna": self.rna[idx],
            "protein": self.protein[idx],
        }

        if self.text is not None:
            item["text"] = self.text[idx]

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx]).float()

        return item
