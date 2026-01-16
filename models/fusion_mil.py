# fusion_mil.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionMIL(nn.Module):
    def __init__(self, dDNA, dRNA, dProt, d_model, d_attn):
        super().__init__()
        self.proj_DNA  = nn.Linear(dDNA,  d_model)
        self.proj_RNA  = nn.Linear(dRNA,  d_model)
        self.proj_Prot = nn.Linear(dProt, d_model)

        self.V_DNA  = nn.Linear(d_model, d_attn)
        self.U_DNA  = nn.Linear(d_model, d_attn)
        self.V_RNA  = nn.Linear(d_model, d_attn)
        self.U_RNA  = nn.Linear(d_model, d_attn)
        self.V_Prot = nn.Linear(d_model, d_attn)
        self.U_Prot = nn.Linear(d_model, d_attn)

        self.W = nn.Linear(d_attn, 1, bias=False)
        self.output_dim = d_model

    def _masked_mean(self, H, mask):
        """
        H:    [B, T, d]
        mask: [B, T] bool
        returns: [B, d]
        """
        m = mask.unsqueeze(-1).float()                 # [B, T, 1]
        summed = (H * m).sum(dim=1)                    # [B, d]
        denom = m.sum(dim=1).clamp(min=1.0)            # [B, 1]
        return summed / denom

    def gated_attention_score(self, h_mean, V, U):
        """
        h_mean: [B, d_model]
        returns score: [B, 1]
        """
        v_term = torch.tanh(V(h_mean))        # [B, d_attn]
        u_term = torch.sigmoid(U(h_mean))     # [B, d_attn]
        gated  = v_term * u_term              # [B, d_attn]
        return self.W(gated)                  # [B, 1]

    def forward(self, DNA, RNA, Prot, mask):
        """
        DNA/RNA/Prot: [B, T, d*]
        mask:         [B, T] bool  (aligned_mask)
        returns: Z_fused [B, T, d_model]
        """
        if mask is None:
            raise ValueError("FusionMIL requires a mask for correct pooling.")

        H_DNA  = self.proj_DNA(DNA)     # [B, T, d_model]
        H_RNA  = self.proj_RNA(RNA)
        H_Prot = self.proj_Prot(Prot)

        h_DNA  = self._masked_mean(H_DNA,  mask)  # [B, d_model]
        h_RNA  = self._masked_mean(H_RNA,  mask)
        h_Prot = self._masked_mean(H_Prot, mask)

        s_DNA  = self.gated_attention_score(h_DNA,  self.V_DNA,  self.U_DNA)   # [B,1]
        s_RNA  = self.gated_attention_score(h_RNA,  self.V_RNA,  self.U_RNA)
        s_Prot = self.gated_attention_score(h_Prot, self.V_Prot, self.U_Prot)

        scores = torch.cat([s_DNA, s_RNA, s_Prot], dim=1)   # [B, 3]
        alpha = F.softmax(scores, dim=1)                    # [B, 3]

        a0 = alpha[:, 0].view(-1, 1, 1)
        a1 = alpha[:, 1].view(-1, 1, 1)
        a2 = alpha[:, 2].view(-1, 1, 1)

        Z_fused = a0 * H_DNA + a1 * H_RNA + a2 * H_Prot     # [B, T, d_model]
        Z_fused = Z_fused * mask.unsqueeze(-1).float()

        return Z_fused, alpha
