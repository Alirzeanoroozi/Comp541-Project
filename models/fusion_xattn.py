import torch
import torch.nn as nn

class FusionCrossAttention(nn.Module):
    def __init__(self, dDNA, dRNA, dProt, d_model, num_heads):
        super().__init__()
        self.proj_DNA  = nn.Linear(dDNA,  d_model)
        self.proj_RNA  = nn.Linear(dRNA,  d_model)
        self.proj_Prot = nn.Linear(dProt, d_model)

        self.attn_DNA  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_RNA  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_Prot = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.out_DNA  = nn.Linear(d_model, d_model)
        self.out_RNA  = nn.Linear(d_model, d_model)
        self.out_Prot = nn.Linear(d_model, d_model)

        self.fusion_proj = nn.Linear(3 * d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.output_dim = d_model

    def forward(self, DNA, RNA, Prot, mask):
        if mask is None:
            raise ValueError("FusionCrossAttention requires a mask for key padding.")

        mask = mask.to(DNA.device)
        if mask.dtype != torch.bool:
            mask = mask.bool()

        H_DNA  = self.proj_DNA(DNA)
        H_RNA  = self.proj_RNA(RNA)
        H_Prot = self.proj_Prot(Prot)

        C = torch.cat([H_DNA, H_RNA, H_Prot], dim=1)   # [B, 3T, d_model]
        c_kpm = (~mask).repeat(1, 3)                   # [B, 3T] True=ignore

        Z_DNA,  _ = self.attn_DNA (H_DNA,  C, C, key_padding_mask=c_kpm, need_weights=False)
        Z_RNA,  _ = self.attn_RNA (H_RNA,  C, C, key_padding_mask=c_kpm, need_weights=False)
        Z_Prot, _ = self.attn_Prot(H_Prot, C, C, key_padding_mask=c_kpm, need_weights=False)

        Z_DNA  = self.out_DNA(Z_DNA)
        Z_RNA  = self.out_RNA(Z_RNA)
        Z_Prot = self.out_Prot(Z_Prot)

        Z_concat = torch.cat([Z_DNA, Z_RNA, Z_Prot], dim=-1)
        Z = self.fusion_proj(Z_concat)
        Z_resid = (Z_DNA + Z_RNA + Z_Prot) / 3

        Z_fused = self.layernorm(Z + Z_resid)
        Z_fused = Z_fused * mask.unsqueeze(-1).float()
        return Z_fused

