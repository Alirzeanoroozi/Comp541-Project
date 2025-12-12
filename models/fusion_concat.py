# fusion_concat.py

import torch
import torch.nn as nn

class FusionConcat(nn.Module):
    """
    Concatenation fusion from BioLangFusion (Section 2.2.1).

    Given aligned embeddings:
        DNA:  (T', dDNA)
        RNA:  (T', dRNA)
        Prot: (T', dProt)

    Output:
        Z_concat: (T', dDNA' + dRNA + dProt)
    
    Paper notes:
    - DNA embedding dimension is much larger than RNA/Protein.
      To avoid dominance, apply a learnable MLP to reduce dDNA → dDNA'.
    """

    def __init__(self, dDNA, dRNA, dProt, dDNA_proj):
        """
        dDNA: original DNA dim from LM (e.g., 2560)
        dRNA: RNA dim from LM
        dProt: protein dim
        dDNA_proj: desired projected DNA dimension (paper uses smaller than others)
        """
        super().__init__()

        # Project DNA dimension
        self.dna_mlp = nn.Sequential(
            nn.Linear(dDNA, dDNA_proj),
            nn.ReLU(),
            nn.Linear(dDNA_proj, dDNA_proj)
        )

        self.output_dim = dDNA_proj + dRNA + dProt

    def forward(self, DNA, RNA, Prot):
        """
        DNA, RNA, Prot already aligned (T', d_m).

        Returns:
            fused: (T', output_dim)
        """

        # Project DNA first
        proj_DNA = self.dna_mlp(DNA)  # (T', dDNA_proj)

        # Concatenate along feature dimension
        fused = torch.cat([proj_DNA, RNA, Prot], dim=-1)

        return fused
