import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        task: str,
        num_classes: int = None,
        proj_dim: int = 640,
        kernel_sizes=(3, 4, 5),
        out_channels: int = 100,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")

        if task == "classification" and num_classes is None:
            raise ValueError("num_classes must be provided for classification")

        self.task = task

        self.project = nn.Linear(embed_dim, proj_dim)

        self.kernel_sizes = list(kernel_sizes)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(proj_dim, out_channels, kernel_size=k, padding=k // 2)
                for k in self.kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)

        out_dim = num_classes if task == "classification" else 1
        self.fc = nn.Linear(len(self.kernel_sizes) * out_channels, out_dim)

        if activation.lower() == "relu":
            self.act = F.relu
        elif activation.lower() == "tanh":
            self.act = torch.tanh
        elif activation.lower() == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"Unknown activation '{activation}'")

    @staticmethod
    def _conv1d_out_len(L_in, k, pad, stride=1, dilation=1):
        return (L_in + 2 * pad - dilation * (k - 1) - 1) // stride + 1

    def forward(self, x, mask=None):
        x = self.project(x)
        x = x.transpose(1, 2)

        feats = []
        L = x.size(-1)

        for conv, k in zip(self.convs, self.kernel_sizes):
            c = self.act(conv(x))
            if mask is not None:
                pad = conv.padding[0]
                stride = conv.stride[0]
                dilation = conv.dilation[0]
                Lk = c.size(-1)

                expected = self._conv1d_out_len(L, k, pad, stride, dilation)
                m = mask

                if expected > m.size(1):
                    m = F.pad(m, (0, expected - m.size(1)), value=False)
                elif expected < m.size(1):
                    m = m[:, :expected]

                if m.size(1) > Lk:
                    m = m[:, :Lk]
                elif m.size(1) < Lk:
                    m = F.pad(m, (0, Lk - m.size(1)), value=False)

                m = m.unsqueeze(1)
                c = c.masked_fill(~m, float("-inf"))

            feats.append(c.max(dim=2).values)

        feat = torch.cat(feats, dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)
