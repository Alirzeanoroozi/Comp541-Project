import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNHead(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Project to 640-dim space
        self.project = nn.Linear(embed_dim, 640)

        # Conv1d expects input: (B, C_in, seq_len)
        self.conv1 = nn.Conv1d(640, 100, kernel_size=3, padding=1)  # L_out = L
        self.conv2 = nn.Conv1d(640, 100, kernel_size=4, padding=2)  # L_out = L + 1
        self.conv3 = nn.Conv1d(640, 100, kernel_size=5, padding=2)  # L_out = L

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(3 * 100, 1)

    def forward(self, x, mask=None):
        """
        x:    (B, L, D)
        mask: (B, L) bool, True for real tokens, False for padding
        """
        x = self.project(x)      # (B, L, 640)
        x = x.transpose(1, 2)    # (B, 640, L)

        c1 = F.relu(self.conv1(x))   # (B, 100, L)
        c2 = F.relu(self.conv2(x))   # (B, 100, L+1)
        c3 = F.relu(self.conv3(x))   # (B, 100, L)

        if mask is not None:
            # mask -> broadcastable shape for conv outputs
            # conv1/conv3 keep length L
            m1 = mask.unsqueeze(1)  # (B, 1, L)
            m3 = m1

            # conv2 outputs length L+1 due to padding=2, k=4
            # simplest correct alignment: pad mask with one False on the right
            m2 = F.pad(mask, (0, 1), value=False).unsqueeze(1)  # (B, 1, L+1)

            # Set padded positions to -inf so max-pool ignores them
            c1 = c1.masked_fill(~m1, float("-inf"))
            c2 = c2.masked_fill(~m2, float("-inf"))
            c3 = c3.masked_fill(~m3, float("-inf"))

        # Max-pool over temporal dimension
        c1 = c1.max(dim=2).values  # (B, 100)
        c2 = c2.max(dim=2).values  # (B, 100)
        c3 = c3.max(dim=2).values  # (B, 100)

        feat = torch.cat([c1, c2, c3], dim=1)  # (B, 300)
        feat = self.dropout(feat)
        return self.fc(feat)  # (B, 1)
