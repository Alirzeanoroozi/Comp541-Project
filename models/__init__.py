from .dna_model import NucleotideTransformerEmbedder
from .rna_model import RNAFMEmbedder
from .protein_model import ESM2Embedder
#from .text_model import TextEmbedder

from .projection_heads import ProjectionHead
from .prediction_head import TextCNNHead, MLPHead

from .fusion_concat import FusionConcat
from .fusion_mil import FusionMIL
from .fusion_xattn import FusionCrossAttention

#from .lora_adapter import LoRAAdapter


from .unimodel import build_unimodal_model
from .multi_modal import build_multimodal_model


def build_model(config):
    """
    Unified model factory.
    """
    modality = config.get("modality")

    if modality is None:
        raise ValueError("config must contain 'modality'")

    if modality.lower() == "multi":
        return build_multimodal_model(config)
    else:
        return build_unimodal_model(config)
