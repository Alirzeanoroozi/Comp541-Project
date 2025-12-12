# fusion_mil.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionMIL(nn.Module):
    """
    Modality-level MIL fusion with gated attention.
    Implements BioLangFusion Section 2.2.2 exactly.

    Inputs (aligned):
        DNA:  (T', dDNA)
        RNA:  (T', dRNA)
        Prot: (T', dProt)

    Steps:
        1) Project all modalities to shared dim d
        2) Mean-pool each modality → h̄_m
        3) Compute gated attention score:
               score_m = Wᵀ [ tanh(V_m h̄_m + b_m)  ⊙  σ(U_m h̄_m + c_m) ]
        4) Convert to α_m via softmax
        5) Fused = Σ α_m * H_m
    """

    def __init__(self, dDNA, dRNA, dProt, d_model, d_attn):
        """
        dDNA, dRNA, dProt : input dims of DNA/RNA/Protein embeddings
        d_model           : shared projected dimension H_m ∈ ℝ^{T' × d_model}
        d_attn            : dimension of the gating networks (from paper)
        """
        super().__init__()

        # ------- (1) Project modalities to shared dim d_model -------
        self.proj_DNA  = nn.Linear(dDNA,  d_model)
        self.proj_RNA  = nn.Linear(dRNA,  d_model)
        self.proj_Prot = nn.Linear(dProt, d_model)

        # ------- (2) Attention parameters for each modality -------
        # V_m, U_m are modality-specific (BioLangFusion eq. 2)
        self.V_DNA  = nn.Linear(d_model, d_attn)
        self.U_DNA  = nn.Linear(d_model, d_attn)
        self.V_RNA  = nn.Linear(d_model, d_attn)
        self.U_RNA  = nn.Linear(d_model, d_attn)
        self.V_Prot = nn.Linear(d_model, d_attn)
        self.U_Prot = nn.Linear(d_model, d_attn)

        # Shared attention vector W (d_attn → 1)
        self.W = nn.Linear(d_attn, 1, bias=False)

    def gated_attention_score(self, h_mean, V, U):
        """
        Computes score_m = Wᵀ [ tanh(V h̄_m) ⊙ σ(U h̄_m) ].
        """
        v_term = torch.tanh(V(h_mean))      # (d_attn)
        u_term = torch.sigmoid(U(h_mean))   # (d_attn)
        gated  = v_term * u_term            # element-wise ⊙
        score  = self.W(gated)              # scalar
        return score

    def forward(self, DNA, RNA, Prot):
        """
        Inputs: (T', d_m)
        Output: Z_fused of shape (T', d_model)
        """

        # -------- (1) Project modalities to shared representation H_m --------
        H_DNA  = self.proj_DNA(DNA)     # (T', d_model)
        H_RNA  = self.proj_RNA(RNA)
        H_Prot = self.proj_Prot(Prot)

        # -------- (2) Mean-pool each modality --------
        h_DNA  = H_DNA.mean(dim=0)      # (d_model)
        h_RNA  = H_RNA.mean(dim=0)
        h_Prot = H_Prot.mean(dim=0)

        # -------- (3) Compute gated attention logits via Eq. (2) --------
        score_DNA  = self.gated_attention_score(h_DNA,  self.V_DNA,  self.U_DNA)
        score_RNA  = self.gated_attention_score(h_RNA,  self.V_RNA,  self.U_RNA)
        score_Prot = self.gated_attention_score(h_Prot, self.V_Prot, self.U_Prot)

        scores = torch.stack([score_DNA, score_RNA, score_Prot], dim=0)  # (3,1)

        # -------- (4) Softmax over modalities --------
        alpha = F.softmax(scores.squeeze(-1), dim=0)  # (3)

        # -------- (5) Fused representation --------
        Z_fused = alpha[0] * H_DNA + alpha[1] * H_RNA + alpha[2] * H_Prot

        return Z_fused      # shape: (T', d_model), as in paper
