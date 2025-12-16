import torch.nn as nn
from models.prediction_head import TextCNNHead

class UnimodalRegressionModel(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = TextCNNHead(input_dim)
    
    def forward(self, seqs, embeddings):  
        return self.net(embeddings).squeeze(-1)  # (B,)
    
def build_model(config):
    return UnimodalRegressionModel(input_dim=config['embedding_dim'])