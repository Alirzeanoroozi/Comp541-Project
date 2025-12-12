import torch
import torch.nn as nn
from models.rna_model import RNAFMEmbedder

class BatchMLPHead(nn.Module):
    """Batch-compatible MLP head for regression."""
    def __init__(self, input_dim=256, num_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (B, D) - already batched and pooled
        return self.net(x).squeeze(-1)  # (B,)


class RNARegressionModel(nn.Module):
    """Simple RNA regression model combining embedder and prediction head."""
    
    def __init__(self, embedder, prediction_head):
        super().__init__()
        self.embedder = embedder
        self.prediction_head = prediction_head
    
    def forward(self, rna_seqs):
        # Handle both single string and list of strings
        if isinstance(rna_seqs, str):
            rna_seqs = [rna_seqs]
        elif isinstance(rna_seqs, list):
            pass  # Already a list
        else:
            # Assume it's a batch from DataLoader (list of strings)
            rna_seqs = list(rna_seqs) if not isinstance(rna_seqs, list) else rna_seqs
        
        # Process each sequence and stack embeddings
        batch_embeddings = []
        for seq in rna_seqs:
            embeddings = self.embedder(seq)  # (L, D)
            # Average pool over sequence length to get fixed-size representation
            pooled = embeddings.mean(dim=0)  # (D,)
            batch_embeddings.append(pooled)
        
        # Stack to form batch
        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # (B, D)
        
        # Get predictions
        predictions = self.prediction_head(batch_embeddings)  # (B,)
        return predictions
    

def build_model(config):
    embedder = RNAFMEmbedder(max_len=config['max_len'], device=config['device'])
    prediction_head = BatchMLPHead(input_dim=config['projection_dim'], num_classes=1)
    return RNARegressionModel(embedder, prediction_head)