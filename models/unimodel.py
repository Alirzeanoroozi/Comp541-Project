import torch.nn as nn
from models.prediction_head import TextCNNHead

class UnimodalRegressionModel(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = TextCNNHead(input_dim)

    def forward(self, embeddings, mask=None):
        # embeddings: [B, L, D], mask: [B, L] (bool)
        return self.net(embeddings, mask=mask).squeeze(-1)

def build_model(config):
    return UnimodalRegressionModel(input_dim=config['embedding_dim'])
