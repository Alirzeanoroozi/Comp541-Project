import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ESM2Embedder(nn.Module):
    def __init__(self, device, max_len):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
        self.model.eval()
        
    def forward(self, seq):
        tokens = self.tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens)
        return out.last_hidden_state.squeeze(0)

