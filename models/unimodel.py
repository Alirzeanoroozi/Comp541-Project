import torch
import torch.nn as nn
from models.rna_model import RNAFMEmbedder
from models.protein_model import ESM2Embedder
from models.prediction_head import MLPHead

class UnimodalRegressionModel(nn.Module):
    def __init__(self, embedder, prediction_head):
        super().__init__()
        self.embedder = embedder
        self.prediction_head = prediction_head
    
    def forward(self, seqs):        
        batch_embeddings = []
        for seq in seqs:
            embeddings = self.embedder(seq)  # (L, D)
            pooled = embeddings.mean(dim=0)  # (D,)
            batch_embeddings.append(pooled)
        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # (B, D)
        predictions = self.prediction_head(batch_embeddings)  # (B,)
        return predictions
    
def build_model(config):
    if config['modality'] == 'RNA':
        embedder = RNAFMEmbedder(max_len=config['max_len'], device=config['device'])
    elif config['modality'] == 'Protein':
        embedder = ESM2Embedder(max_len=config['max_len'], device=config['device'])
    else:
        raise ValueError(f"Invalid modality: {config['modality']}")
    prediction_head = MLPHead(input_dim=config['embedding_dim'], projection_dim=config['projection_dim'])
    return UnimodalRegressionModel(embedder, prediction_head)