import torch
import torch.nn as nn
from typing import List, Sequence, Tuple, Union, Optional

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, TaskType
import fm

from models.prediction_head import TextCNNHead


BatchSeq = Union[List[str], Tuple[str, ...], Sequence[str]]


class ProteinEncoder(nn.Module):
    """
    ESM2 encoder for protein sequences with LoRA fine-tuning.
    Returns token-level representations and an attention mask.
    """

    def __init__(self, device: str, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        base_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense"],  # Common transformer modules
            bias="none",
        )

        # Apply LoRA to the model
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)

        tok_max = getattr(self.tokenizer, "model_max_length", None)
        self.max_tokens = int(tok_max) if tok_max and tok_max < 10**6 else 1024

        # Hidden size used by downstream head
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))

    def forward(self, seqs: BatchSeq) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seqs: list / sequence of protein strings
        Returns:
          embeddings: [B, L, D]
          mask:       [B, L] bool (True = real token)
        """
        if isinstance(seqs, (list, tuple)):
            texts = [str(s) for s in seqs]
        else:
            texts = [str(s) for s in seqs]

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        out = self.model(**tokens)
        embeddings = out.last_hidden_state  # [B, L, D]
        mask = tokens.get("attention_mask", torch.ones(embeddings.shape[:2], device=self.device)).bool()
        return embeddings, mask


class DNAEncoder(nn.Module):
    """
    Nucleotide Transformer encoder for DNA sequences with LoRA fine-tuning.
    Uses the final hidden state from AutoModelForMaskedLM.
    """

    def __init__(self, device: str, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
            trust_remote_code=True,
        )
        base_model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
            trust_remote_code=True,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            bias="none",
        )

        # Apply LoRA to the model
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)

        tok_max = getattr(self.tokenizer, "model_max_length", None)
        self.max_tokens = int(tok_max) if tok_max and tok_max < 10**6 else 1024

        self.hidden_size = int(getattr(self.model.config, "hidden_size"))

    def forward(self, seqs: BatchSeq) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seqs: list / sequence of DNA strings
        Returns:
          embeddings: [B, L, D]
          mask:       [B, L] bool
        """
        if isinstance(seqs, (list, tuple)):
            texts = [str(s) for s in seqs]
        else:
            texts = [str(s) for s in seqs]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get(
            "attention_mask",
            (input_ids != self.tokenizer.pad_token_id),
        ).to(self.device)

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        embeddings = out.hidden_states[-1]  # [B, L, D]
        mask = attention_mask.bool()
        return embeddings, mask


class RNAEncoder(nn.Module):
    """
    RNA-FM encoder for RNA sequences.
    Note: RNA-FM is not a HuggingFace model, so we use standard fine-tuning
    (not PEFT LoRA). Only the encoder layers are trainable.
    Uses layer 12 representations.
    """

    def __init__(self, device: str):
        super().__init__()
        self.device = device

        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

        # Freeze embedding layer, make encoder trainable
        if hasattr(self.model, "embed_tokens"):
            for param in self.model.embed_tokens.parameters():
                param.requires_grad = False

        # Make encoder layers trainable for fine-tuning
        if hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = True

        # Best-effort hidden size extraction; falls back to 768 if unavailable.
        hidden_size = None
        if hasattr(self.model, "embed_tokens"):
            emb = getattr(self.model.embed_tokens, "embedding_dim", None)
            hidden_size = int(emb) if emb is not None else None
        if hidden_size is None and hasattr(self.model, "encoder"):
            hidden_size = int(getattr(self.model.encoder, "embed_dim", 768))
        if hidden_size is None:
            hidden_size = 768
        self.hidden_size = hidden_size

    def forward(self, seqs: BatchSeq) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seqs: list / sequence of RNA strings
        Returns:
          embeddings: [B, L, D]
          mask:       [B, L] bool
        """
        if isinstance(seqs, (list, tuple)):
            texts = [str(s) for s in seqs]
        else:
            texts = [str(s) for s in seqs]

        # RNA-FM expects a list of (id, sequence) tuples
        data = [(f"seq_{i}", s) for i, s in enumerate(texts)]
        _, _, batch_tokens = self.batch_converter(data)  # [B, L]
        batch_tokens = batch_tokens.to(self.device)

        # Mask: non-padding tokens
        pad_idx = self.alphabet.padding_idx
        mask = (batch_tokens != pad_idx)

        results = self.model(batch_tokens, repr_layers=[12])
        embeddings = results["representations"][12]  # [B, L, D]

        return embeddings, mask


class LoRARegressionModel(nn.Module):
    """
    Sequence-to-scalar regression model with LoRA fine-tuning for a single modality.
    
    For Protein (ESM2) and DNA (Nucleotide Transformer): Uses PEFT LoRA adapters.
    For RNA (RNA-FM): Uses standard fine-tuning (RNA-FM doesn't support PEFT).
    
    Only LoRA parameters (or encoder layers for RNA) are trainable, making this
    parameter-efficient compared to full fine-tuning.
    """

    def __init__(
        self,
        modality: str,
        device: str = "cpu",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.modality = modality
        self.device = device

        modality_lower = modality.lower()
        if modality_lower == "protein":
            self.encoder = ProteinEncoder(device, lora_r, lora_alpha, lora_dropout)
        elif modality_lower == "dna":
            self.encoder = DNAEncoder(device, lora_r, lora_alpha, lora_dropout)
        elif modality_lower == "rna":
            self.encoder = RNAEncoder(device)
        else:
            raise ValueError(f"Unsupported modality '{modality}'. Use 'RNA', 'DNA', or 'Protein'.")

        # TextCNNHead performs its own projection; we just need the encoder dim.
        self.head = TextCNNHead(embed_dim=self.encoder.hidden_size)

    def forward(self, seqs: BatchSeq):
        """
        seqs: batch of raw sequences as a list/sequence of strings.
        Returns:
          preds: [B] regression scores
        """
        embeddings, mask = self.encoder(seqs)  # [B, L, D], [B, L]
        preds = self.head(embeddings, mask=mask)  # [B, 1]
        return preds.squeeze(-1)


def build_model(config: dict) -> nn.Module:
    """
    Factory used by training scripts.

    Expected keys in config:
      - 'modality': one of 'RNA', 'DNA', 'Protein'
      - 'device':   'cuda' or 'cpu'
      - 'lora_r':   LoRA rank (default: 8)
      - 'lora_alpha': LoRA alpha (default: 16)
      - 'lora_dropout': LoRA dropout (default: 0.1)
    """
    modality = config.get("modality", "RNA")
    device = config.get("device", "cuda")
    lora_r = config.get("lora_r", 8)
    lora_alpha = config.get("lora_alpha", 16)
    lora_dropout = config.get("lora_dropout", 0.1)
    
    model = LoRARegressionModel(
        modality=modality,
        device=device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    return model

