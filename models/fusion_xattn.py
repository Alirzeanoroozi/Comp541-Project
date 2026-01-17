import torch
import torch.nn as nn

class FusionCrossAttention(nn.Module):
    def __init__(
        self,
        dDNA: int,
        dRNA: int,
        dProt: int,
        d_model: int,
        num_heads: int,
        proj_activation: str = "tanh",
        attn_dropout: float = 0.1,
        out_dropout: float = 0.2,
    ):
        super().__init__()

        act = proj_activation.lower()
        if act == "tanh":
            act_layer = nn.Tanh()
        elif act == "relu":
            act_layer = nn.ReLU()
        elif act == "gelu":
            act_layer = nn.GELU()
        elif act in ("none", "linear", ""):
            act_layer = nn.Identity()
        else:
            raise ValueError(f"Unknown proj_activation='{proj_activation}'")

        self.proj_DNA  = nn.Sequential(nn.Linear(dDNA,  d_model), act_layer)
        self.proj_RNA  = nn.Sequential(nn.Linear(dRNA,  d_model), act_layer)
        self.proj_Prot = nn.Sequential(nn.Linear(dProt, d_model), act_layer)

        self.attn_DNA  = nn.MultiheadAttention(d_model, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_RNA  = nn.MultiheadAttention(d_model, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_Prot = nn.MultiheadAttention(d_model, num_heads, dropout=attn_dropout, batch_first=True)

        self.out_DNA  = nn.Linear(d_model, d_model)
        self.out_RNA  = nn.Linear(d_model, d_model)
        self.out_Prot = nn.Linear(d_model, d_model)

        self.fusion_proj = nn.Linear(3 * d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(out_dropout) if out_dropout and out_dropout > 0 else nn.Identity()

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

        C = torch.cat([H_DNA, H_RNA, H_Prot], dim=1)     # [B, 3T, d_model]
        c_kpm = (~mask).repeat(1, 3)                     # [B, 3T] True=ignore

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
        Z_fused = self.dropout(Z_fused)
        Z_fused = Z_fused * mask.unsqueeze(-1).float()
        return Z_fused

