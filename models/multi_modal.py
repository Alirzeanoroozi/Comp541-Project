import torch.nn as nn
import torch

from .dna_model import NucleotideTransformerEmbedder
from .rna_model import RNAFMEmbedder
from .protein_model import ESM2Embedder

from .projection_heads import ProjectionHead
from .alignment import ModalityAlignment

from .fusion_concat import FusionConcat
from .fusion_mil import FusionMIL
from .fusion_xattn import FusionCrossAttention

from .prediction_head import MLPHead


class MultiModalFusionModel(nn.Module):
    def __init__(self, encoders, projections, alignment, fusion, head):
        super().__init__()

        self.enc_dna = encoders["dna"]
        self.enc_rna = encoders["rna"]
        self.enc_prot = encoders["protein"]

        self.proj_dna = projections["dna"]
        self.proj_rna = projections["rna"]
        self.proj_prot = projections["protein"]

        self.align = alignment
        self.fusion = fusion
        self.head = head

    def forward(self, batch):
        """
        batch = {
            "dna": list[str],
            "rna": list[str],
            "protein": list[str],
            "label": tensor(B,)
        }
        """

        fused_batch = []

        B = len(batch["dna"])

        for i in range(B):
            dna = batch["dna"][i]
            rna = batch["rna"][i]
            protein = batch["protein"][i]

            # 1. Encode (each returns (L, D))
            EDNA = self.enc_dna(dna)
            ERNA = self.enc_rna(rna)
            EPROT = self.enc_prot(protein)

            # 2. Project
            EDNA = self.proj_dna(EDNA)
            ERNA = self.proj_rna(ERNA)
            EPROT = self.proj_prot(EPROT)

            # 3. Align
            ADNA, ARNA, APROT = self.align(EDNA, ERNA, EPROT)

            # 4. Fuse
            Z = self.fusion(ADNA, ARNA, APROT)   # (T', D_fused)

            # 5. Pool
            Z = Z.mean(dim=0)                    # (D_fused,)

            fused_batch.append(Z)

        fused_batch = torch.stack(fused_batch, dim=0)  # (B, D_fused)

        # 6. Predict
        return self.head(fused_batch)



def build_multimodal_model(config):

    # -----------------------
    # ENCODERS
    # -----------------------
    encoders = {
        "dna": NucleotideTransformerEmbedder(
            max_len=config["max_len"],
            device=config["device"]
        ),
        "rna": RNAFMEmbedder(
            max_len=config["max_len"],
            device=config["device"]
        ),
        "protein": ESM2Embedder(
            max_len=config["max_len"],
            device=config["device"]
        ),
    }

    # -----------------------
    # FREEZE ENCODERS
    # -----------------------
    if config.get("freeze_encoders", True):
        for enc in encoders.values():
            for p in enc.parameters():
                p.requires_grad = False

    # -----------------------
    # PROJECTIONS
    # -----------------------
    projections = {
        "dna": ProjectionHead(config["dDNA"], config["projection_dim"]),
        "rna": ProjectionHead(config["dRNA"], config["projection_dim"]),
        "protein": ProjectionHead(config["dProt"], config["projection_dim"]),
    }

    # -----------------------
    # ALIGNMENT
    # -----------------------
    alignment = ModalityAlignment(
        dDNA=config["projection_dim"],
        dRNA=config["projection_dim"],
        dProt=config["projection_dim"],
    )

    # -----------------------
    # FUSION
    # -----------------------
    
    fusion_type = config.get("fusion_type", "concat")

    if fusion_type == "concat":
        fusion = FusionConcat(
            dDNA=config["projection_dim"],
            dRNA=config["projection_dim"],
            dProt=config["projection_dim"],
            dDNA_proj=config["projection_dim"],
        )
        fusion_out_dim = fusion.output_dim

    elif fusion_type == "mil":
        fusion = FusionMIL(config["projection_dim"])
        fusion_out_dim = fusion.output_dim

    elif fusion_type == "xattn":
        fusion = FusionCrossAttention(
            dDNA=config["projection_dim"],
            dRNA=config["projection_dim"],
            dProt=config["projection_dim"],
            d_model=config["projection_dim"],
            num_heads=config.get("num_heads", 4),
        )
        fusion_out_dim = fusion.output_dim
        
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    # -----------------------
    # HEAD
    # -----------------------
    head = MLPHead(
        input_dim=fusion_out_dim,
        projection_dim=config["projection_dim"],
        num_classes=1
    )

    return MultiModalFusionModel(
        encoders=encoders,
        projections=projections,
        alignment=alignment,
        fusion=fusion,
        head=head
    )
