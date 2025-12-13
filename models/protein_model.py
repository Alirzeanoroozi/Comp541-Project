import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ESM2Embedder(nn.Module):
    def __init__(self, max_len=512, device="cpu"):
        super().__init__()
        self.device = device
        self.max_len = max_len

        self.model_name = "facebook/esm2_t30_150M_UR50D"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, seq):
        try:
            tokens = self.tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len).to(self.device)
            out = self.model(**tokens)
            emb = out.last_hidden_state.squeeze(0)
            print(emb.shape)
            return emb
        except Exception as e:
            print(f"Error: {e}")
            print(f"Sequence: {seq}")
            return None
