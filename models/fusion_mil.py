import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionMIL(nn.Module):
    """
    MIL fusion with entropy regularization.
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, dna_emb, rna_emb, protein_emb):
        # Stack: (3, L, d)
        stacked = torch.stack([dna_emb, rna_emb, protein_emb], dim=0)

        # Compute attention per modality
        scores = self.attn(stacked).squeeze(-1)  # (3, L)
        weights = F.softmax(scores.mean(dim=1), dim=0)  # (3,)

        # Weighted sum
        fused = (weights[0] * dna_emb +
                 weights[1] * rna_emb +
                 weights[2] * protein_emb)

        return fused   # (L, d)
