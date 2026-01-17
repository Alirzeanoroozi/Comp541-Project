import torch.nn as nn
from models.prediction_head import TextCNNHead

class UnimodalRegressionModel(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = TextCNNHead(input_dim, task="regression",)

    def forward(self, embeddings, mask=None):
        # embeddings: [B, L, D], mask: [B, L] (bool)
        return self.net(embeddings, mask=mask).squeeze(-1)

class UnimodalClassificationModel(nn.Module):
    def __init__(self, input_dim=256, num_classes=3):
        super().__init__()
        self.net = TextCNNHead(
            embed_dim=input_dim,
            task="classification",
            num_classes=num_classes,  
        )

    def forward(self, embeddings, mask=None):
        return self.net(embeddings, mask=mask)

    
    
def build_model(config):
    task = config.get("task", "classification")

    if task == "classification":
        return UnimodalClassificationModel(
            input_dim=config["embedding_dim"],
            num_classes=config["num_classes"]   
        )
    else:
        # 🔥 regression MUST output 1 value
        return UnimodalRegressionModel(
            input_dim=config["embedding_dim"]
        )

