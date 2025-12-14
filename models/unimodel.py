import torch
import torch.nn as nn

class UnimodalRegressionModel(nn.Module):
    def __init__(self, input_dim=256, projection_dim=600, num_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, num_classes)
        )
    
    def forward(self, seqs, embeddings):  
        # embeddings: (B, L, D) --> (B, D)
        pooled_embeddings = []
        for seq, embedding in zip(seqs, embeddings):
            max_len = max(len(seq), embedding.shape[0])
            embedding = embedding[:max_len].mean(dim=0)  # (D,)
            pooled_embeddings.append(embedding)
        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)  # (B, D)
        return self.net(pooled_embeddings).squeeze(-1)  # (B,)
    
def build_model(config):
    return UnimodalRegressionModel(input_dim=config['embedding_dim'], projection_dim=config['projection_dim'])