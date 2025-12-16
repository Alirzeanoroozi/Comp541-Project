import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ESM2Embedder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        self.model_name = "facebook/esm2_t6_8M_UR50D"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.model.eval()
        
    def forward(self, seq):
        tokens = self.tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens)
        return out.last_hidden_state.squeeze(0)

