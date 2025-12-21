import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

class NucleotideTransformerEmbedder(nn.Module):
    def __init__(self, device, max_len):
        super().__init__()
        self.device = device
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True).to(device)
        self.model.eval()

    def forward(self, seq):
        tokens_ids = self.tokenizer.batch_encode_plus([seq], return_tensors="pt", padding="max_length", max_length = self.max_len)["input_ids"]
        attention_mask = tokens_ids != self.tokenizer.pad_token_id
        tokens_ids = tokens_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            torch_outs = self.model(tokens_ids, attention_mask=attention_mask, encoder_attention_mask=attention_mask, output_hidden_states=True)

        return torch_outs['hidden_states'][-1].squeeze(0)