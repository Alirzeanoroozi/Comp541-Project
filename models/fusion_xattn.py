# fusion_xattn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionCrossAttention(nn.Module):
    
    """
    Cross-modal Multi-Head Attention Fusion
    Implements BioLangFusion Section 2.2.3 exactly.

    Inputs (aligned):
        DNA:  (T', dDNA)
        RNA:  (T', dRNA)
        Prot: (T', dProt)

    Output:
        Z_fused: (T', d_model) — final fused representation.
    """
    
    def __init__(self, dDNA, dRNA, dProt, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.output_dim = d_model   # ✅ REQUIRED LINE

        # -------------------------------------------------------------
        # (1) Project modalities to shared space H_m ∈ R^{T' × d_model}
        # -------------------------------------------------------------
        self.proj_DNA  = nn.Linear(dDNA,  d_model)
        self.proj_RNA  = nn.Linear(dRNA,  d_model)
        self.proj_Prot = nn.Linear(dProt, d_model)

        # -------------------------------------------------------------
        # (2) Multi-head attention for each modality query
        #     Each modality has its own Q/K/V projections (paper Eq. 4)
        # -------------------------------------------------------------
        self.attn_DNA  = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.attn_RNA  = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.attn_Prot = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        # -------------------------------------------------------------
        # Output projections after attention (part of g() in paper Eq. 4)
        # -------------------------------------------------------------
        self.out_DNA  = nn.Linear(d_model, d_model)
        self.out_RNA  = nn.Linear(d_model, d_model)
        self.out_Prot = nn.Linear(d_model, d_model)

        # -------------------------------------------------------------
        # (3) Fusion projection for concatenated Z_m outputs
        # -------------------------------------------------------------
        self.fusion_proj = nn.Linear(3 * d_model, d_model)

        # (4) Final layer normalization (Eq. 5)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, DNA, RNA, Prot):
        """
        Inputs:
            DNA, RNA, Prot: (T', d_m)

        Returns:
            Z_fused: (T', d_model)
        """

        # -----------------------------
        # (1) Project to shared space
        # -----------------------------
        H_DNA  = self.proj_DNA(DNA)
        H_RNA  = self.proj_RNA(RNA)
        H_Prot = self.proj_Prot(Prot)

        # -----------------------------
        # (2) Build context C = [DNA; RNA; Prot]
        # -----------------------------
        C = torch.cat([H_DNA, H_RNA, H_Prot], dim=0)  # shape (3T', d_model)

        # Expand to batch before attention
        C_batch = C.unsqueeze(0)

        # Queries for each modality
        Q_DNA  = H_DNA.unsqueeze(0)
        Q_RNA  = H_RNA.unsqueeze(0)
        Q_Prot = H_Prot.unsqueeze(0)

        # -----------------------------
        # (3) Cross-modal attention (Eq. 4)
        # -----------------------------
        Z_DNA, _  = self.attn_DNA (Q_DNA,  C_batch, C_batch)
        Z_RNA, _  = self.attn_RNA (Q_RNA,  C_batch, C_batch)
        Z_Prot, _ = self.attn_Prot(Q_Prot, C_batch, C_batch)

        # Remove batch dimension
        Z_DNA  = self.out_DNA(Z_DNA.squeeze(0))
        Z_RNA  = self.out_RNA(Z_RNA.squeeze(0))
        Z_Prot = self.out_Prot(Z_Prot.squeeze(0))

        # -----------------------------
        # (4) Concatenate and project (paper Eq. 5)
        # -----------------------------
        Z_concat = torch.cat([Z_DNA, Z_RNA, Z_Prot], dim=-1)  # (T', 3*d_model)
        Z = self.fusion_proj(Z_concat)                        # (T', d_model)

        # Modality-average residual term
        Z_resid = (Z_DNA + Z_RNA + Z_Prot) / 3

        # -----------------------------
        # (5) Final representation (Eq. 5)
        # -----------------------------
        Z_fused = self.layernorm(Z + Z_resid)

        return Z_fused
