import torch.nn as nn

from models.alignment import ModalityAlignmentBatch
from models.prediction_head import TextCNNHead
from models.fusion_concat import FusionConcat
from models.fusion_mil import FusionMIL
from models.fusion_xattn import FusionCrossAttention


class MultimodalModel(nn.Module):
    def __init__(
        self,
        dDNA: int,
        dRNA: int,
        dProt: int,
        task: str,
        num_classes: int,
        fusion_type: str = "concat",

        dna_up_kernel_size: int = 3,
        dna_up_stride: int = 2,
        dna_up_padding: int = 2,
        dna_up_output_padding: int = 0,
        dna_up_bias: bool = False,
        rna_pool_kernel_size: int = 3,
        rna_pool_stride: int = 3,

        d_model: int = 600,
        d_attn: int = 100,
        num_heads: int = 4,

        mil_proj_activation: str = "tanh",
        tau_init: float = 1.0,
        tau_min: float = 0.02,
        tau_max: float = 20.0,

        xattn_proj_activation: str = "tanh",
        attn_dropout: float = 0.1,
        out_dropout: float = 0.2,

        head_proj_dim: int = 640,
        head_kernel_sizes=(3, 4, 5),
        head_out_channels: int = 100,
        head_dropout: float = 0.2,
        head_activation: str = "relu",
    ):
        super().__init__()   

        # ---------------- Task ----------------
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")

        if task == "classification" and num_classes is None:
            raise ValueError("num_classes required for classification")
        self.task = task
        # ---------------- Alignment ----------------
        self.alignment = ModalityAlignmentBatch(
            dDNA=dDNA,
            dRNA=dRNA,
            dProt=dProt,
            dna_up_kernel_size=dna_up_kernel_size,
            dna_up_stride=dna_up_stride,
            dna_up_padding=dna_up_padding,
            dna_up_output_padding=dna_up_output_padding,
            dna_up_bias=dna_up_bias,
            rna_pool_kernel_size=rna_pool_kernel_size,
            rna_pool_stride=rna_pool_stride,
        )
        # ---------------- Fusion ----------------
        fusion_type = fusion_type.lower()
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.fusion = FusionConcat(dDNA=dDNA, dRNA=dRNA, dProt=dProt)
            fused_dim = self.fusion.output_dim

        elif fusion_type == "mil":
            self.fusion = FusionMIL(
                dDNA=dDNA,
                dRNA=dRNA,
                dProt=dProt,
                d_model=d_model,
                d_attn=d_attn,
                proj_activation=mil_proj_activation,
                tau_init=tau_init,
                tau_min=tau_min,
                tau_max=tau_max,
            )
            fused_dim = self.fusion.output_dim

        elif fusion_type == "xattn":
            self.fusion = FusionCrossAttention(
                dDNA=dDNA,
                dRNA=dRNA,
                dProt=dProt,
                d_model=d_model,
                num_heads=num_heads,
                proj_activation=xattn_proj_activation,
                attn_dropout=attn_dropout,
                out_dropout=out_dropout,
            )
            fused_dim = self.fusion.output_dim

        else:
            raise ValueError(f"Unknown fusion_type='{fusion_type}'. Use one of: concat, mil, xattn")

        # ---------------- Head ----------------
        self.head = TextCNNHead(
            embed_dim=fused_dim,
            task=task,                       
            num_classes=num_classes,         
            proj_dim=head_proj_dim,
            kernel_sizes=head_kernel_sizes,
            out_channels=head_out_channels,
            dropout=head_dropout,
            activation=head_activation,
        )



    def forward(self, dna_pad, rna_pad, prot_pad, dna_len, rna_len, prot_len):
        dna_a, rna_a, prot_a, _, aligned_mask = self.alignment(
            dna_pad, rna_pad, prot_pad, dna_len, rna_len, prot_len
        )

        if self.fusion_type == "mil":
            fused, alpha = self.fusion(dna_a, rna_a, prot_a, mask=aligned_mask)
            out = self.head(fused, mask=aligned_mask)
            if self.task == "regression":
                out = out.squeeze(-1)
            return out, {"alpha": alpha}

        fused = self.fusion(dna_a, rna_a, prot_a, mask=aligned_mask)
        out = self.head(fused, mask=aligned_mask)
        if self.task == "regression":
            out = out.squeeze(-1)
        return out



def build_model(config: dict):
    """
    Required:
      dDNA, dRNA, dProt
      fusion_type in {concat,mil,xattn}

    Optional (Alignment):
      dna_up_kernel_size, dna_up_stride, dna_up_padding, dna_up_output_padding, dna_up_bias
      rna_pool_kernel_size, rna_pool_stride

    Optional (Fusion):
      d_model, d_attn, num_heads
      mil_proj_activation, tau_init, tau_min, tau_max
      xattn_proj_activation, attn_dropout, out_dropout

    Optional (Head):
      head_proj_dim, head_kernel_sizes, head_out_channels, head_dropout, head_activation
    """
    num_classes = config.get("num_classes")
    head_kernel_sizes = config.get("head_kernel_sizes", config.get("kernel_sizes", (3, 4, 5)))

    return MultimodalModel(
        dDNA=config["dDNA"],
        dRNA=config["dRNA"],
        dProt=config["dProt"],
        task=config["task"],  
        fusion_type=config.get("fusion_type", "concat"),
        num_classes=num_classes,

        # alignment
        dna_up_kernel_size=config.get("dna_up_kernel_size", 3),
        dna_up_stride=config.get("dna_up_stride", 2),
        dna_up_padding=config.get("dna_up_padding", 2),
        dna_up_output_padding=config.get("dna_up_output_padding", 0),
        dna_up_bias=config.get("dna_up_bias", False),
        rna_pool_kernel_size=config.get("rna_pool_kernel_size", 3),
        rna_pool_stride=config.get("rna_pool_stride", 3),

        # fusion shared
        d_model=config.get("d_model", 600),
        d_attn=config.get("d_attn", 100),
        num_heads=config.get("num_heads", 4),

        # MIL
        mil_proj_activation=config.get("mil_proj_activation", "tanh"),
        tau_init=config.get("tau_init", 1.0),
        tau_min=config.get("tau_min", 0.02),
        tau_max=config.get("tau_max", 20.0),

        # XAttn
        xattn_proj_activation=config.get("xattn_proj_activation", "tanh"),
        attn_dropout=config.get("attn_dropout", 0.1),
        out_dropout=config.get("out_dropout", 0.2),

        # head
        head_proj_dim=config.get("head_proj_dim", 640),
        head_kernel_sizes=head_kernel_sizes,
        head_out_channels=config.get("head_out_channels", 100),
        head_dropout=config.get("head_dropout", 0.2),
        head_activation=config.get("head_activation", "relu"),
    )

