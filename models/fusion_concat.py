import torch
import torch.nn as nn


class FusionConcat(nn.Module):
    """
    Simplified concatenation fusion: concat feature dimension.
    """

    def __init__(self):
        super().__init__()

    def forward(self, dna_emb, rna_emb, protein_emb):
        # Truncate to shortest length for alignment
        L = min(len(dna_emb), len(rna_emb), len(protein_emb))
        dna_emb = dna_emb[:L]
        rna_emb = rna_emb[:L]
        protein_emb = protein_emb[:L]

        return torch.cat([dna_emb, rna_emb, protein_emb], dim=-1)
