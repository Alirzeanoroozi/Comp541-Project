import torch
import torch.nn as nn
import fm

class RNAFMEmbedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.model.eval()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, seq):
        seq = str(seq)

        data = [("RNA", seq)]
        _, _, batch_tokens = self.batch_converter(data)

        with torch.no_grad():
            results = self.model(batch_tokens.to(self.device), repr_layers=[12])

        return results["representations"][12].squeeze(0)  # (tokens, dim)
