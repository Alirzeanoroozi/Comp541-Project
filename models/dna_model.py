import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

class NucleotideTransformerEmbedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True).to(device)
        self.model.eval()

        # nucleotide transformer max_tokens safety check
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        self.max_tokens = int(tok_max) if tok_max and tok_max < 10**6 else 1024

    def forward(self, seq):
        seq = str(seq)
        enc = self.tokenizer([seq], return_tensors="pt", padding=False,truncation=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", (input_ids != self.tokenizer.pad_token_id)).to(self.device)

        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        return out.hidden_states[-1].squeeze(0)  # (tokens, dim)
