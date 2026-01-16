import torch
import torch.nn as nn

class FusionConcat(nn.Module):
    def __init__(self, dDNA: int, dRNA: int, dProt: int):
        super().__init__()
        self.output_dim = dDNA + dRNA + dProt

    def forward(self, DNA, RNA, Prot, mask=None):
        """
        DNA, RNA, Prot: [B, T, d*]
        mask (optional): [B, T] bool
        returns: fused [B, T, output_dim]
        """
        if not (
            DNA.size(0) == RNA.size(0) == Prot.size(0)
            and DNA.size(1) == RNA.size(1) == Prot.size(1)
        ):
            raise ValueError(f"Shape mismatch: DNA={DNA.shape}, RNA={RNA.shape}, Prot={Prot.shape}")

        fused = torch.cat([DNA, RNA, Prot], dim=-1)  # [B, T, dDNA+dRNA+dProt]

        if mask is not None:
            fused = fused * mask.unsqueeze(-1).float()

        return fused

