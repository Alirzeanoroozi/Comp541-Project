import torch
import torch.nn as nn
import fm

class RNAFMEmbedder(nn.Module):
    def __init__(self, max_len, device):
        super().__init__()
        self.max_len = max_len
        self.device = device

        # Load RNA-FM model and alphabet from the fm package
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

        # Freeze weights
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, seq):
        data = [("RNA", seq[:self.max_len])]
        _, _, batch_tokens = self.batch_converter(data)

        with torch.no_grad():
            results = self.model(batch_tokens.to(self.device), repr_layers=[12])

        emb = results["representations"][12].squeeze(0)  # (L, D)

        # If length < max_len, pad to max_len
        L, D = emb.shape
        if L < self.max_len:
            pad = torch.zeros(self.max_len - L, D, device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=0)
        elif L > self.max_len:
            emb = emb[:self.max_len, :]

        return emb  # (max_len, embedding_dim)
