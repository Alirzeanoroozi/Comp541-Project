import torch.nn as nn
from models.alignment import ModalityAlignmentBatch
from models.prediction_head import TextCNNHead
from models.fusion_concat import FusionConcat
from models.fusion_mil import FusionMIL
from models.fusion_xattn import FusionCrossAttention


class MultimodalRegressionModel(nn.Module):
    """
    Forward expects batched padded embeddings + their lengths/masks:

      dna_pad:  [B, Ld, dDNA]
      rna_pad:  [B, Lr, dRNA]
      prot_pad: [B, Lp, dProt]
      dna_len, rna_len, prot_len: [B]
      (optional) dna_mask/rna_mask/prot_mask not needed by alignment_batch

    Internally:
      - align -> [B, T, d*] + aligned_mask [B, T]
      - fuse  -> [B, T, D_fused]
      - TextCNNHead -> [B, 1] -> squeeze -> [B]
    """

    def __init__(self,
        dDNA: int,
        dRNA: int,
        dProt: int,
        fusion_type: str = "concat",
        # concat-only
        dDNA_proj: int = 256,
        # mil/xattn shared projection
        d_model: int = 256,
        # mil
        d_attn: int = 128,
        # xattn
        num_heads: int = 4,
    ):
        super().__init__()
        # alignment
        self.alignment = ModalityAlignmentBatch(dDNA=dDNA, dRNA=dRNA, dProt=dProt)

        # fusion
        fusion_type = fusion_type.lower()
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.fusion = FusionConcat(dDNA=dDNA, dRNA=dRNA, dProt=dProt)
            fused_dim = self.fusion.output_dim
        elif fusion_type == "mil":
            self.fusion = FusionMIL(dDNA=dDNA, dRNA=dRNA, dProt=dProt, d_model=d_model, d_attn=d_attn)
            fused_dim = self.fusion.output_dim
        elif fusion_type == "xattn":
            self.fusion = FusionCrossAttention(dDNA=dDNA, dRNA=dRNA, dProt=dProt, d_model=d_model, num_heads=num_heads)
            fused_dim = self.fusion.output_dim
        else:
            raise ValueError(f"Unknown fusion_type='{fusion_type}'. Use one of: concat, mil, xattn")

        # prediction head
        self.head = TextCNNHead(embed_dim=fused_dim)

    def forward(self, dna_pad, rna_pad, prot_pad, dna_len, rna_len, prot_len):
        dna_a, rna_a, prot_a, aligned_len, aligned_mask = self.alignment(dna_pad, rna_pad, prot_pad, dna_len, rna_len, prot_len)

        # different forward implementation if using MIL (returns attention weights)
        if self.fusion_type == "mil":
            fused, alpha = self.fusion(dna_a, rna_a, prot_a, mask=aligned_mask)
            out = self.head(fused, mask=aligned_mask).squeeze(-1)  # [B]
            return out, {"alpha": alpha}
        else:
            fused = self.fusion(dna_a, rna_a, prot_a, mask=aligned_mask)
            out = self.head(fused, mask=aligned_mask).squeeze(-1)  # [B]
            return out


def build_model(config):
    """
    Expected config keys (suggested):
      config["dDNA"], config["dRNA"], config["dProt"]
      config["fusion_type"] in {"concat","mil","xattn"}

    Optional:
      config["dDNA_proj"], config["d_model"], config["d_attn"], config["num_heads"]
    """
    return MultimodalRegressionModel(
        dDNA=config["dDNA"],
        dRNA=config["dRNA"],
        dProt=config["dProt"],
        fusion_type=config.get("fusion_type", "concat"),
        dDNA_proj=config.get("dDNA_proj", 256),
        d_model=config.get("d_model", 256),
        d_attn=config.get("d_attn", 128),
        num_heads=config.get("num_heads", 4),
    )
