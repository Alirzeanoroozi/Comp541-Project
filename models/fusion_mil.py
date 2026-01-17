import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionMIL(nn.Module):
    def __init__(
        self,
        dDNA: int,
        dRNA: int,
        dProt: int,
        d_model: int,
        d_attn: int,
        proj_activation: str = "tanh",
        tau_init: float = 1.0,
        tau_min: float = 0.02,
        tau_max: float = 20.0,
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

        # Per-modality projection + activation
        self.proj_DNA  = nn.Sequential(nn.Linear(dDNA,  d_model), act_layer)
        self.proj_RNA  = nn.Sequential(nn.Linear(dRNA,  d_model), act_layer)
        self.proj_Prot = nn.Sequential(nn.Linear(dProt, d_model), act_layer)

        # Gated attention
        self.V_DNA  = nn.Linear(d_model, d_attn)
        self.U_DNA  = nn.Linear(d_model, d_attn)
        self.V_RNA  = nn.Linear(d_model, d_attn)
        self.U_RNA  = nn.Linear(d_model, d_attn)
        self.V_Prot = nn.Linear(d_model, d_attn)
        self.U_Prot = nn.Linear(d_model, d_attn)

        self.W = nn.Linear(d_attn, 1, bias=False)

        # Learned softmax temperature
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        self.output_dim = d_model

    def _masked_mean(self, H, mask):
        m = mask.unsqueeze(-1).float()              # [B, T, 1]
        summed = (H * m).sum(dim=1)                 # [B, d]
        denom = m.sum(dim=1).clamp(min=1.0)         # [B, 1]
        return summed / denom

    def gated_attention_score(self, h_mean, V, U):
        v_term = torch.tanh(V(h_mean))              # [B, d_attn]
        u_term = torch.sigmoid(U(h_mean))           # [B, d_attn]
        gated  = v_term * u_term                    # [B, d_attn]
        return self.W(gated)                        # [B, 1]

    def forward(self, DNA, RNA, Prot, mask):
        if mask is None:
            raise ValueError("FusionMIL requires a mask for correct pooling.")

        if mask.dtype != torch.bool:
            mask = mask.bool()
        mask = mask.to(DNA.device)

        H_DNA  = self.proj_DNA(DNA)                 # [B, T, d_model]
        H_RNA  = self.proj_RNA(RNA)
        H_Prot = self.proj_Prot(Prot)

        h_DNA  = self._masked_mean(H_DNA,  mask)    # [B, d_model]
        h_RNA  = self._masked_mean(H_RNA,  mask)
        h_Prot = self._masked_mean(H_Prot, mask)

        s_DNA  = self.gated_attention_score(h_DNA,  self.V_DNA,  self.U_DNA)   # [B,1]
        s_RNA  = self.gated_attention_score(h_RNA,  self.V_RNA,  self.U_RNA)
        s_Prot = self.gated_attention_score(h_Prot, self.V_Prot, self.U_Prot)

        scores = torch.cat([s_DNA, s_RNA, s_Prot], dim=1)   # [B, 3]

        tau = self.tau.clamp(self.tau_min, self.tau_max)
        alpha = F.softmax(scores / tau, dim=1)              # [B, 3]

        a0 = alpha[:, 0].view(-1, 1, 1)
        a1 = alpha[:, 1].view(-1, 1, 1)
        a2 = alpha[:, 2].view(-1, 1, 1)

        Z_fused = a0 * H_DNA + a1 * H_RNA + a2 * H_Prot     # [B, T, d_model]
        Z_fused = Z_fused * mask.unsqueeze(-1).float()

        return Z_fused, alpha

