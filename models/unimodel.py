import torch
import torch.nn as nn

from models.rna_model import RNAFMEmbedder
from models.protein_model import ESM2Embedder
from models.dna_model import NucleotideTransformerEmbedder

from models.prediction_head import MLPHead

class UnimodalRegressionModel(nn.Module):
    def __init__(self, embedder, prediction_head):
        super().__init__()
        self.embedder = embedder
        self.prediction_head = prediction_head
    
    def forward(self, seqs):        
        batch_embeddings = []
        for seq in seqs:
            seq = str(seq)
            embeddings = self.embedder(seq)  # (L, D)
            pooled = embeddings.mean(dim=0)  # (D,)
            batch_embeddings.append(pooled)
        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # (B, D)
        predictions = self.prediction_head(batch_embeddings)  # (B,)
        return predictions


def build_unimodal_model(config):
    modality = config["modality"].lower()

    if modality == "dna":
        embedder = NucleotideTransformerEmbedder(
            max_len=config["max_len"],
            device=config["device"]
        )
        embedding_dim = 768   # ✅ DNABERT-2

    elif modality == "rna":
        embedder = RNAFMEmbedder(
            max_len=config["max_len"],
            device=config["device"]
        )
        embedding_dim = 640   # ✅ RNA-FM

    elif modality == "protein":
        embedder = ESM2Embedder(
            max_len=config["max_len"],
            device=config["device"]
        )
        embedding_dim = 640   # ✅ ESM2

    else:
        raise ValueError(f"Unknown unimodal modality: {modality}")

    prediction_head = MLPHead(
        input_dim=embedding_dim,              # ✅ correct
        projection_dim=config["projection_dim"]
    )

    return UnimodalRegressionModel(embedder, prediction_head)
