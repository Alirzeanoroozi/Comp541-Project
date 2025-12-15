#projection_heads,py

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Projects each modality embedding into shared latent space D.
    """

    def __init__(self, input_dim, output_dim=256):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.proj(x)    # (L, D)
