import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ESM2Embedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(self.device)
        self.model.eval()

        # ESM2 max_tokens safety check
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        self.max_tokens = int(tok_max) if tok_max and tok_max < 10**6 else 1024

    def forward(self, seq):
        seq = str(seq)

        tokens = self.tokenizer(seq, return_tensors="pt", padding=False, truncation=True, max_length=self.max_tokens)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.model(**tokens)

        return out.last_hidden_state.squeeze(0)  # (tokens, dim)
