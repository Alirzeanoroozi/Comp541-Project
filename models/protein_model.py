import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ESM2Embedder(nn.Module):
    def __init__(self, max_len=512, device="cpu"):
        super().__init__()
        self.device = device
        self.max_len = max_len

        self.model_name = "facebook/esm2_t30_150M_UR50D"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, seq):
        tokens = self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=False      # ✅ NO padding
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**tokens)

        emb = out.last_hidden_state.squeeze(0)  # (L, D)

        # Optional: remove BOS/EOS
        if emb.shape[0] >= 2:
            emb = emb[1:-1]

        return emb