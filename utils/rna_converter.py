from typing import Optional, Tuple
from pathlib import Path
import argparse
import pandas as pd

""""Adds DNA and Protein columns to CSV, renames the 'Sequence' column to 'RNA'"""

GENETIC_CODE = {
    "UUU": "F", "UUC": "F",
    "UUA": "L", "UUG": "L", "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I",
    "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S", "AGU": "S", "AGC": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y",
    "CAU": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C",
    "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "UAA": "*", "UAG": "*", "UGA": "*",
}


def clean_mrna(seq: str) -> str:
    """Uppercase and keep only A/C/G/U (convert T → U)."""
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace("T", "U")
    return "".join(base for base in seq if base in {"A", "C", "G", "U"})


def mrna_to_dna(mrna: str) -> str:
    """Convert mRNA to DNA (U → T)."""
    mrna = clean_mrna(mrna)
    return mrna.replace("U", "T")


def translate_mrna(mrna: str, frame: int = 0, stop_at_stop: bool = True) -> str:
    """Translate assuming correct frame."""
    mrna = clean_mrna(mrna)
    aa_seq = []
    for i in range(frame, len(mrna) - 2, 3):
        codon = mrna[i:i+3]
        aa = GENETIC_CODE.get(codon, "X")
        if aa == "*" and stop_at_stop:
            break
        aa_seq.append(aa)
    return "".join(aa_seq)


def find_longest_orf(mrna: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Find the longest start→stop ORF."""
    mrna = clean_mrna(mrna)
    stop_codons = {"UAA", "UAG", "UGA"}
    best = (None, None, None)
    best_len = 0

    for frame in (0, 1, 2):
        i = frame
        while i < len(mrna) - 2:
            if mrna[i:i+3] == "AUG":
                j = i + 3
                while j < len(mrna) - 2:
                    if mrna[j:j+3] in stop_codons:
                        length = j + 3 - i
                        if length > best_len:
                            best_len = length
                            best = (frame, i, j + 3)
                        break
                    j += 3
                i = j
            else:
                i += 3
    return best


def translate_longest_orf(mrna: str) -> Optional[str]:
    frame, start, end = find_longest_orf(mrna)
    if frame is None:
        return None
    coding_region = clean_mrna(mrna)[start:end]
    return translate_mrna(coding_region, frame=0, stop_at_stop=True)

def process_csv_file(path: Path) -> None:
    """
    Reads CSV:
        - must contain column "Sequence"
    Writes:
        <stem>_multimodal.csv
    with columns:
        DNA, Protein, RNA, <all original columns except old Sequence>
    """
    print(f"Processing: {path}")

    df = pd.read_csv(path)

    if "Sequence" not in df.columns:
        raise ValueError(f"No 'Sequence' column found in {path}")

    # Rename Sequence → RNA (cleaned RNA)
    df = df.rename(columns={"Sequence": "RNA"})
    df["RNA"] = df["RNA"].apply(clean_mrna)

    # New DNA column
    df["DNA"] = df["RNA"].apply(mrna_to_dna)

    # New Protein column
    df["Protein"] = df["RNA"].apply(lambda s: translate_longest_orf(s) or "")

    # Reorder columns: DNA, Protein, RNA, then all the others
    cols = ["DNA", "Protein", "RNA"] + [c for c in df.columns if c not in {"DNA", "Protein", "RNA"}]
    df = df[cols]

    out_path = path.with_name(f"{path.stem}_multimodal.csv")
    df.to_csv(out_path, index=False)

    print(f"  -> wrote {out_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Convert mRNA CSVs to multimodal CSVs.")
    parser.add_argument("input_dir", type=str, help="Folder containing .csv files.")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_path in csv_files:
        process_csv_file(csv_path)


if __name__ == "__main__":
    main()
