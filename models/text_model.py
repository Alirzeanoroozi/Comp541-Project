import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEmbedder(nn.Module):
    def __init__(self, max_len=128, device="cpu"):
        super().__init__()
        self.device = device
        self.max_len = max_len

        self.model_name = "dmis-lab/biobert-base-cased-v1.1"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)

        # Freeze BioBERT weights
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, text):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )

        # Move each tensor to the device
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.model(**tokens)

        # shape: (1, seq_len, hidden_dim)
        emb = out.last_hidden_state

        # shape: (seq_len, hidden_dim)
        return emb.squeeze(0)
