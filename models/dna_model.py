import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class NucleotideTransformerEmbedder(nn.Module):
    def __init__(self, model_name="zhihan1996/DNABERT-2-117M", max_len=512, device="cpu"):
        super().__init__()

        print(f"Loading public DNA model: {model_name}")

        self.device = device
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, seq):
        tokens = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**tokens)

        # DNABERT-2 returns tuple, not ModelOutput
        if isinstance(out, tuple):
            hidden = out[0]                 # (1, L, D)
        else:
            hidden = out.last_hidden_state  # fallback (never hurts)

        hidden = hidden.squeeze(0)          # (L, D)

        return hidden


