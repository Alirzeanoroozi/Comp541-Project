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
    
    def forward(self, embeddings):  
         
        batch_embeddings = []
        for embedding in embeddings:
            pooled = embedding.mean(dim=0)  # (D,)
            batch_embeddings.append(pooled)
        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # (B, D)
        # x: (B, D) - already batched and pooled
        return self.net(batch_embeddings).squeeze(-1)  # (B,)
    
def build_model(config):
    return UnimodalRegressionModel(input_dim=config['embedding_dim'], projection_dim=config['projection_dim'])