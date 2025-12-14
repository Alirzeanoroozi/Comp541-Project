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
        # embeddings: (B, L, D) --> (B, D)
        pooled_embeddings = embeddings.mean(dim=1)  # (B, D)
        return self.net(pooled_embeddings).squeeze(-1)  # (B,)
    
def build_model(config):
    return UnimodalRegressionModel(input_dim=config['embedding_dim'], projection_dim=config['projection_dim'])