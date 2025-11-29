import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """
    Applies LoRA to any linear layer.
    """

    def __init__(self, linear_layer: nn.Linear, rank=8, alpha=16):
        super().__init__()

        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha

        # Low-rank matrices
        self.A = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.B = nn.Linear(rank, linear_layer.out_features, bias=False)

        # Scale
        self.scaling = alpha / rank

        # Freeze original layer
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        update = self.B(self.A(x)) * self.scaling
        return base + update
