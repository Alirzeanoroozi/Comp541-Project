import torch
import torch.nn as nn


class FusionCrossAttention(nn.Module):
    """
    Cross-modal multi-head attention fusion.
    """

    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, dna_emb, rna_emb, protein_emb):
        # Stack sequences along temporal axis
        joint = torch.cat([dna_emb, rna_emb, protein_emb], dim=0).unsqueeze(0)

        # Self-attention
        attn_out, _ = self.attn(joint, joint, joint)

        # Residual + layer norm
        fused = self.ln(attn_out + joint).squeeze(0)

        return fused
