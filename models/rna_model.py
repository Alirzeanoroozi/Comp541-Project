import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class RNAFMEmbedder(nn.Module):
    def __init__(self, max_len=512, device="cpu"):
        super().__init__()

        self.max_len = max_len
        self.device = device

        # SAFE, WORKING MODEL: facebook/esm2_t6_8M_UR50D
        self.model_name = "facebook/esm2_t6_8M_UR50D"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(self.device)

        # Freeze weights
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, seq: str):
        """Returns per-token embeddings padded/truncated to max_len."""

        tokens = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        ).to(self.device)

        outputs = self.model(**tokens)
        emb = outputs.last_hidden_state.squeeze(0)  # (L, D)

        return emb  # already (max_len, embedding_dim)
