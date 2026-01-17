import torch
import torch.nn as nn


class FusionCrossAttention(nn.Module):
    """
      1) Project each modality to shared dim d_model (with optional activation)
      2) Build context C = [H_DNA, H_RNA, H_Prot] along time (B, 3T, d_model)
      3) For each modality m:
           A_m = MHA(Q=H_m, K=C, V=C)
           Z_m = LN_m( H_m + Dropout( Proj_m(A_m) ) )    # g(.) includes projection + residual + LayerNorm
      4) Concatenate Z_m across feature dim -> Linear fusion -> residual avg + final LayerNorm
      5) Apply output dropout and mask
    """

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
        use_per_modality_residual_ln: bool = True,
        use_final_residual_avg: bool = True,
        use_final_layernorm: bool = True,
    ):
        super().__init__()

        # projection activation
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

        # per-modality projection to d_model
        self.proj_DNA = nn.Sequential(nn.Linear(dDNA, d_model), act_layer)
        self.proj_RNA = nn.Sequential(nn.Linear(dRNA, d_model), act_layer)
        self.proj_Prot = nn.Sequential(nn.Linear(dProt, d_model), act_layer)

        # per-modality attention
        self.attn_DNA = nn.MultiheadAttention(d_model, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_RNA = nn.MultiheadAttention(d_model, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_Prot = nn.MultiheadAttention(d_model, num_heads, dropout=attn_dropout, batch_first=True)

        # g(.) projection layers
        self.out_DNA = nn.Linear(d_model, d_model)
        self.out_RNA = nn.Linear(d_model, d_model)
        self.out_Prot = nn.Linear(d_model, d_model)

        # g(.) residual + layernorm per modality
        self.use_per_modality_residual_ln = bool(use_per_modality_residual_ln)
        self.ln_DNA = nn.LayerNorm(d_model)
        self.ln_RNA = nn.LayerNorm(d_model)
        self.ln_Prot = nn.LayerNorm(d_model)

        # fusion head
        self.fusion_proj = nn.Linear(3 * d_model, d_model)
        self.use_final_residual_avg = bool(use_final_residual_avg)
        self.use_final_layernorm = bool(use_final_layernorm)
        self.final_ln = nn.LayerNorm(d_model) if self.use_final_layernorm else nn.Identity()

        self.dropout = nn.Dropout(out_dropout) if out_dropout and out_dropout > 0 else nn.Identity()
        self.output_dim = d_model

    def _g(self, H, A, out_proj, ln):
        """
        g(.) block per modality: projection + (optional) residual + LayerNorm
          H: [B, T, d]
          A: [B, T, d] attention output
        """
        Z = out_proj(A)                  # projection
        if self.use_per_modality_residual_ln:
            Z = ln(H + Z)                # residual + LayerNorm
        return Z

    def forward(self, DNA, RNA, Prot, mask):
        """
        DNA/RNA/Prot: [B, T, d*]
        mask:         [B, T] bool (aligned_mask). True=real, False=pad
        returns:      [B, T, d_model]
        """
        if mask is None:
            raise ValueError("FusionCrossAttention requires a mask for key padding.")

        device = DNA.device
        mask = mask.to(device)
        if mask.dtype != torch.bool:
            mask = mask.bool()

        # Project to shared dim
        H_DNA = self.proj_DNA(DNA)       # [B, T, d]
        H_RNA = self.proj_RNA(RNA)
        H_Prot = self.proj_Prot(Prot)

        # context C = concat along time: [B, 3T, d]
        C = torch.cat([H_DNA, H_RNA, H_Prot], dim=1)

        # key_padding_mask expects True for positions to ignore
        c_kpm = (~mask).repeat(1, 3)     # [B, 3T]

        # cross-modal attention
        A_DNA, _ = self.attn_DNA(H_DNA, C, C, key_padding_mask=c_kpm, need_weights=False)
        A_RNA, _ = self.attn_RNA(H_RNA, C, C, key_padding_mask=c_kpm, need_weights=False)
        A_Prot, _ = self.attn_Prot(H_Prot, C, C, key_padding_mask=c_kpm, need_weights=False)

        # g(.) per modality: projection + residual + LN
        Z_DNA = self._g(H_DNA, A_DNA, self.out_DNA, self.ln_DNA)     # [B, T, d]
        Z_RNA = self._g(H_RNA, A_RNA, self.out_RNA, self.ln_RNA)
        Z_Prot = self._g(H_Prot, A_Prot, self.out_Prot, self.ln_Prot)

        # fusion: concat streams -> linear
        Z_concat = torch.cat([Z_DNA, Z_RNA, Z_Prot], dim=-1)         # [B, T, 3d]
        Z = self.fusion_proj(Z_concat)                               # [B, T, d]

        # final residual avg + final LN
        if self.use_final_residual_avg:
            Z_resid = (Z_DNA + Z_RNA + Z_Prot) / 3.0
            Z = Z + Z_resid

        Z_fused = self.final_ln(Z)
        Z_fused = self.dropout(Z_fused)

        # mask padding positions
        Z_fused = Z_fused * mask.unsqueeze(-1).float()
        return Z_fused


