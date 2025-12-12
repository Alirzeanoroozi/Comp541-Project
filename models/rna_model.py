import torch
import torch.nn as nn
import fm

# # 1. Load RNA-FM model
# model, alphabet = fm.pretrained.rna_fm_t12()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# batch_converter = alphabet.get_batch_converter()
# model.eval()  # disables dropout for deterministic results

# # 2. Prepare data
# data = [
#     ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
#     ("RNA2", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
#     ("RNA3", "CGAUUCNCGUUCCC--CCGCCUCCA"),
# ]
# batch_labels, batch_strs, batch_tokens = batch_converter(data)

# # 3. Extract embeddings (on CPU)
# with torch.no_grad():
#     results = model(batch_tokens.to(device), repr_layers=[12])
# token_embeddings = results["representations"][12]

# print(token_embeddings.shape)

# print(token_embeddings[0])

class RNAFMEmbedder(nn.Module):
    def __init__(self, max_len=512, device="cpu"):
        super().__init__()
        self.max_len = max_len
        self.device = device

        # Load RNA-FM model and alphabet from the fm package
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

        # Freeze weights
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, seq: str):
        """Returns per-token embeddings padded/truncated to max_len using RNA-FM."""

        # RNA-FM expects a batch of data in (label, sequence) tuples even for single sequence
        data = [("RNA", seq[:self.max_len])]  # truncate to max_len if needed
        _, _, batch_tokens = self.batch_converter(data)

        with torch.no_grad():
            results = self.model(batch_tokens.to(self.device), repr_layers=[12])

        emb = results["representations"][12].squeeze(0)  # (L, D)

        # If length < max_len, pad to max_len
        L, D = emb.shape
        if L < self.max_len:
            pad = torch.zeros(self.max_len - L, D, device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=0)
        elif L > self.max_len:
            emb = emb[:self.max_len, :]

        return emb  # (max_len, embedding_dim)
