# alignment.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityAlignment(nn.Module):
    """
    Implements the codon-level modality alignment described in BioLangFusion (Section 2.1).
    
    Inputs:
        EDNA:  (T/6, dDNA)  from 6-mer tokenizer (Nucleotide Transformer)
        ERNA:  (T,   dRNA)  from nucleotide tokenizer (RNA-FM)
        EProt: (T/3, dProt) from amino acid tokenizer (ESM-2)
    
    Output:
        aligned_DNA, aligned_RNA, aligned_Prot  all ∈ (T', d_m)
        where T' = T/3 (codon resolution)
    """

    def __init__(self, dDNA, dRNA, dProt):
        super().__init__()

        # Paper: TransposedConv for DNA (upsample 6-mer tokens to codon level)
        # We use ConvTranspose1d(kernel=2, stride=2) as described.
        self.dna_up = nn.ConvTranspose1d(
            in_channels=dDNA,
            out_channels=dDNA,
            kernel_size=2,
            stride=2,
            bias=False
        )

        # Paper: RNA downsampled with non-overlapping AvgPool k=3, s=3
        self.rna_down = nn.AvgPool1d(kernel_size=3, stride=3)

        # Protein is already at codon resolution → no projection needed
        self.identity = nn.Identity()

    def forward(self, EDNA, ERNA, EProt):
        """
        EDNA:  (TDNA, dDNA)
        ERNA:  (TRNA, dRNA)
        EProt: (TProt, dProt)
        """

        # --- DNA Upsampling ---
        # EDNA: (T6, dDNA) → need (B, C, L)
        dna = EDNA.transpose(0,1).unsqueeze(0)  # (1, dDNA, T6)
        dna_up = self.dna_up(dna)               # (1, dDNA, T')
        aligned_DNA = dna_up.squeeze(0).transpose(0,1)  # (T', dDNA)

        # --- RNA Downsampling ---
        # ERNA: (T, dRNA) → (1, dRNA, T)
        rna = ERNA.transpose(0,1).unsqueeze(0)
        rna_down = self.rna_down(rna)                   # (1, dRNA, T')
        aligned_RNA = rna_down.squeeze(0).transpose(0,1)

        # --- Protein stays the same ---
        aligned_Prot = EProt  # Already (T', dProt)

        # Safety check: enforce equal sequence length
        T_min = min(len(aligned_DNA), len(aligned_RNA), len(aligned_Prot))

        aligned_DNA  = aligned_DNA[:T_min]
        aligned_RNA  = aligned_RNA[:T_min]
        aligned_Prot = aligned_Prot[:T_min]

        return aligned_DNA, aligned_RNA, aligned_Prot

class ModalityAlignmentBatch(nn.Module):
    def __init__(self, dDNA, dRNA, dProt):
        super().__init__()
        # paper: kernel=3, stride=2, padding=2
        self.dna_up = nn.ConvTranspose1d(
            dDNA, dDNA, kernel_size=3, stride=2, padding=2, bias=False
        )
        self.rna_down = nn.AvgPool1d(kernel_size=3, stride=3)

    def forward(self, dna, rna, prot, dna_len, rna_len, prot_len):
        B = dna.size(0)

        dna_up = self.dna_up(dna.transpose(1, 2)).transpose(1, 2)
        rna_down = self.rna_down(rna.transpose(1, 2)).transpose(1, 2)
        prot_same = prot

        # With kernel=3,stride=2,pad=2, output length is: L_out = 2*L_in - 3
        dna_a_len  = (dna_len * 2 - 3).clamp(min=0)
        rna_a_len  = rna_len // 3
        prot_a_len = prot_len

        aligned_len = torch.min(torch.min(dna_a_len, rna_a_len), prot_a_len)  # [B]
        Tmax = int(aligned_len.max().item()) if B > 0 else 0

        dna_a  = dna_up[:, :Tmax, :]
        rna_a  = rna_down[:, :Tmax, :]
        prot_a = prot_same[:, :Tmax, :]

        aligned_mask = (torch.arange(Tmax, device=aligned_len.device).unsqueeze(0) < aligned_len.unsqueeze(1))
        m = aligned_mask.unsqueeze(-1).float()
        dna_a  = dna_a * m
        rna_a  = rna_a * m
        prot_a = prot_a * m

        return dna_a, rna_a, prot_a, aligned_len, aligned_mask

