import torch
import torch.nn as nn
import fm

class RNAFMEmbedder(nn.Module):
    def __init__(self, max_len, device):
        super().__init__()
        
        # Enforce integer and 1022 token limit
        self.max_len = min(int(max_len), 1022)
        self.device = device
        
        # Load pretrained RNA-FM model
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

        # Freeze model weights
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, seq):

        if not isinstance(seq, str):
            raise TypeError(f"Expected string sequence, got {type(seq)}")

        # Clean RNA characters
        seq = seq.upper().replace("T", "U")
        seq = ''.join(base if base in "ACGU" else "U" for base in seq)

        # Truncate BEFORE tokenization
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]

        # Tokenize
        _, _, batch_tokens = self.batch_converter([("RNA", seq)])
        batch_tokens = batch_tokens.to(self.device)

        # Forward through model
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])

        # (L, D) — NO padding here
        emb = results["representations"][12].squeeze(0)

        return emb

