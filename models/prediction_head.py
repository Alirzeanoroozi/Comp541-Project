import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNHead(nn.Module):
    """
    Mask handling:
      - Supports variable output lengths per kernel due to padding.
      - Computes per-kernel mask length via Conv1d length formula and pads/truncates mask accordingly.
    """

    def __init__(
        self,
        embed_dim: int,
        proj_dim: int = 640,
        kernel_sizes=(3, 4, 5),
        out_channels: int = 100,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.project = nn.Linear(embed_dim, proj_dim)

        self.kernel_sizes = list(kernel_sizes)
        self.out_channels = out_channels

        # Create a conv per kernel size
        self.convs = nn.ModuleList()
        for k in self.kernel_sizes:
            # Keep the same "style" you had: padding roughly preserves length for odd k
            # For even k, this can yield L_out = L+1 (like your k=4, pad=2).
            pad = k // 2
            self.convs.append(nn.Conv1d(proj_dim, out_channels, kernel_size=k, padding=pad))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(self.kernel_sizes) * out_channels, 1)

        if activation.lower() == "relu":
            self.act = F.relu
        elif activation.lower() == "tanh":
            self.act = torch.tanh
        elif activation.lower() == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"Unknown activation='{activation}'. Use relu/tanh/gelu.")

    @staticmethod
    def _conv1d_out_len(L_in: int, k: int, pad: int, stride: int = 1, dilation: int = 1) -> int:
        # floor((L + 2p - d*(k-1) - 1)/s + 1)
        numer = L_in + 2 * pad - dilation * (k - 1) - 1
        return (numer // stride) + 1

    def forward(self, x, mask=None):
        """
        x:    [B, L, D]
        mask: [B, L] bool (True=real, False=pad)
        """
        x = self.project(x)        # [B, L, proj_dim]
        x = x.transpose(1, 2)      # [B, proj_dim, L]

        feats = []
        L = x.size(-1)

        for conv, k in zip(self.convs, self.kernel_sizes):
            c = self.act(conv(x))  # [B, C, Lk]
            if mask is not None:
                pad = conv.padding[0]
                stride = conv.stride[0]
                dilation = conv.dilation[0]
                Lk = c.size(-1)

                # Build a mask for this conv's temporal length.
                expected = self._conv1d_out_len(L, k=k, pad=pad, stride=stride, dilation=dilation)

                m = mask
                if expected > m.size(1):
                    m = F.pad(m, (0, expected - m.size(1)), value=False)
                elif expected < m.size(1):
                    m = m[:, :expected]

                # safeguard: adjust mask to match actual conv output length
                if m.size(1) > Lk:
                    m = m[:, :Lk]
                elif m.size(1) < Lk:
                    m = F.pad(m, (0, Lk - m.size(1)), value=False)

                m = m.unsqueeze(1)  # [B, 1, Lk]
                c = c.masked_fill(~m, float("-inf"))

            feats.append(c.max(dim=2).values)  # [B, C]

        feat = torch.cat(feats, dim=1)        # [B, len(kernels)*C]
        feat = self.dropout(feat)
        return self.fc(feat)                  # [B, 1]

