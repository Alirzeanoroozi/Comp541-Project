import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNHead(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=7, padding=3)

        self.fc = nn.Linear(128 * 3, 1)

    def forward(self, x):
        # x: (B, L, D)
        x = x.transpose(1, 2)

        c1 = F.relu(self.conv1(x)).max(dim=-1)[0]
        c2 = F.relu(self.conv2(x)).max(dim=-1)[0]
        c3 = F.relu(self.conv3(x)).max(dim=-1)[0]

        feat = torch.cat([c1, c2, c3], dim=-1) # (B, 1)

        return self.fc(feat)

