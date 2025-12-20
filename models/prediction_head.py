import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNHead(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Project to 640-dim space
        self.project = nn.Linear(embed_dim, 640)
        # Conv1d expects input: (B, C_in, seq_len)
        self.conv1 = nn.Conv1d(640, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(640, 100, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(640, 100, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(3 * 100, 1)

    def forward(self, x):
        # x: (B, L, D)
        x = self.project(x)                # (B, L, 640)
        x = x.transpose(1, 2)              # (B, 640, L)

        c1 = F.relu(self.conv1(x))         # (B, 100, L)
        c2 = F.relu(self.conv2(x))         # (B, 100, L_out2)
        c3 = F.relu(self.conv3(x))         # (B, 100, L)

        # Max-pool over sequence/temporal dimension
        c1 = torch.max(c1, dim=2)[0]       # (B, 100)
        c2 = torch.max(c2, dim=2)[0]       # (B, 100)
        c3 = torch.max(c3, dim=2)[0]       # (B, 100)

        feat = torch.cat([c1, c2, c3], dim=1)  # (B, 300)
        feat = self.dropout(feat)
        return self.fc(feat)

