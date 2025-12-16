import torch
import torch.nn as nn
from prediction_head import TextCNNHead

class UnimodalRegressionModel(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = TextCNNHead(input_dim)
    
    def forward(self, seqs, embeddings):  
        # embeddings: (B, L, D) --> (B, D)
        # pooled_embeddings = []
        # for seq, embedding in zip(seqs, embeddings):
        #     embedding = embedding[:len(seq)].mean(dim=0)  # (D,)
        #     pooled_embeddings.append(embedding)
        # pooled_embeddings = torch.stack(pooled_embeddings, dim=0)  # (B, D)
        return self.net(embeddings).squeeze(-1)  # (B,)
    
def build_model(config):
    return UnimodalRegressionModel(input_dim=config['embedding_dim'])