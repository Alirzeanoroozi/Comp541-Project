import torch
import torch.nn as nn

class ModalityAlignmentBatch(nn.Module):
    """
    Batched codon-level modality alignment.

    Inputs:
      dna:  [B, Ld, dDNA]   (6-mer tokens)
      rna:  [B, Lr, dRNA]   (1-mer tokens)
      prot: [B, Lp, dProt]  (AA tokens, already ~codon level)
      dna_len, rna_len, prot_len: [B]

    Outputs:
      dna_a, rna_a, prot_a: [B, T, d*] (T = max aligned length in batch)
      aligned_len: [B]
      aligned_mask: [B, T] True for real, False for pad
    """

    def __init__(
        self,
        dDNA: int,
        dRNA: int,
        dProt: int,
        # paper defaults:
        dna_up_kernel_size: int = 3,
        dna_up_stride: int = 2,
        dna_up_padding: int = 2,
        dna_up_output_padding: int = 0,
        dna_up_bias: bool = False,
        rna_pool_kernel_size: int = 3,
        rna_pool_stride: int = 3,
    ):
        super().__init__()

        self.dna_up_kernel_size = int(dna_up_kernel_size)
        self.dna_up_stride = int(dna_up_stride)
        self.dna_up_padding = int(dna_up_padding)
        self.dna_up_output_padding = int(dna_up_output_padding)

        self.dna_up = nn.ConvTranspose1d(
            in_channels=dDNA,
            out_channels=dDNA,
            kernel_size=self.dna_up_kernel_size,
            stride=self.dna_up_stride,
            padding=self.dna_up_padding,
            output_padding=self.dna_up_output_padding,
            bias=dna_up_bias,
        )

        self.rna_down = nn.AvgPool1d(
            kernel_size=int(rna_pool_kernel_size),
            stride=int(rna_pool_stride),
        )

    def _dna_out_len(self, L_in: torch.Tensor) -> torch.Tensor:
        """
        ConvTranspose1d length formula (dilation=1):
          L_out = (L_in - 1)*stride - 2*padding + kernel_size + output_padding
        """
        return (
            (L_in - 1) * self.dna_up_stride
            - 2 * self.dna_up_padding
            + self.dna_up_kernel_size
            + self.dna_up_output_padding
        )

    def forward(self, dna, rna, prot, dna_len, rna_len, prot_len):
        B = dna.size(0)

        # DNA upsample: [B, Ld, d] -> [B, d, Ld] -> up -> [B, d, Ld_out] -> [B, Ld_out, d]
        dna_up = self.dna_up(dna.transpose(1, 2)).transpose(1, 2)

        # RNA downsample: [B, Lr, d] -> [B, d, Lr] -> pool -> [B, d, floor(Lr/3)] -> [B, Lr', d]
        rna_down = self.rna_down(rna.transpose(1, 2)).transpose(1, 2)

        # Protein unchanged
        prot_same = prot

        # Valid aligned lengths per sample
        dna_a_len = self._dna_out_len(dna_len).clamp(min=0)
        rna_a_len = (rna_len // 3).clamp(min=0)
        prot_a_len = prot_len.clamp(min=0)

        aligned_len = torch.min(torch.min(dna_a_len, rna_a_len), prot_a_len)  # [B]
        Tmax = int(aligned_len.max().item()) if B > 0 else 0

        # Truncate to Tmax (common batch time dim)
        dna_a = dna_up[:, :Tmax, :]
        rna_a = rna_down[:, :Tmax, :]
        prot_a = prot_same[:, :Tmax, :]

        # Build aligned mask and zero out invalid positions
        aligned_mask = (
            torch.arange(Tmax, device=aligned_len.device).unsqueeze(0)
            < aligned_len.unsqueeze(1)
        )  # [B, T]

        m = aligned_mask.unsqueeze(-1).float()
        dna_a = dna_a * m
        rna_a = rna_a * m
        prot_a = prot_a * m

        return dna_a, rna_a, prot_a, aligned_len, aligned_mask


