from typing import Optional, Tuple
from pathlib import Path
import argparse
import pandas as pd

""""Adds DNA and Protein columns to CSV, renames the 'Sequence' column to 'RNA'.
Supports two translation modes:
    normal mode: longest ORF (requires AUG -> stop)
    direct mode: frame-0 translation ignoring start/stop codons
"""

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


# cleaning if needed
def clean_mrna(seq: str) -> str:
    """Uppercase and keep only A/C/G/U (convert T → U)."""
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace("T", "U")
    return "".join(b for b in seq if b in {"A", "C", "G", "U"})


def mrna_to_dna(mrna: str) -> str:
    return clean_mrna(mrna).replace("U", "T")


# translation functions
def translate_mrna(mrna: str, frame: int = 0, stop_at_stop: bool = True) -> str:
    """Translate in a fixed frame. If stop_at_stop=False, '*' is included."""
    mrna = clean_mrna(mrna)
    aa = []
    for i in range(frame, len(mrna) - 2, 3):
        codon = mrna[i:i+3]
        aa_letter = GENETIC_CODE.get(codon, "X")
        if aa_letter == "*" and stop_at_stop:
            break
        aa.append(aa_letter)
    return "".join(aa)


def find_longest_orf(mrna: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (frame, start, end) for the longest AUG->stop ORF."""
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
                        orf_len = j + 3 - i
                        if orf_len > best_len:
                            best_len = orf_len
                            best = (frame, i, j + 3)
                        break
                    j += 3
                i = j
            else:
                i += 3
    return best


def translate_longest_orf(mrna: str) -> Optional[str]:
    """Translate longest biologically plausible ORF."""
    frame, start, end = find_longest_orf(mrna)
    if frame is None:
        return None
    region = clean_mrna(mrna)[start:end]
    return translate_mrna(region, frame=0, stop_at_stop=True)


# csv processing
def process_csv_file(path: Path, ignore_start_stop: bool) -> None:
    """
    adds DNA + Protein columns, preserves all other columns (Sequence column renamed to RNA)
    translation logic depends on ignore_start_stop:
        False -> longest ORF mode (should start with AUG and end with stop codon)
        True -> no start/stop codon logic i.e. direct frame 0 translation
    """
    print(f"Processing: {path}")

    df = pd.read_csv(path)

    if "Sequence" not in df.columns:
        raise ValueError(f"No 'Sequence' column in {path}")

    # Rename Sequence → RNA
    df = df.rename(columns={"Sequence": "RNA"})
    df["RNA"] = df["RNA"].apply(clean_mrna)

    # DNA
    df["DNA"] = df["RNA"].apply(mrna_to_dna)

    # Protein translation mode
    if ignore_start_stop:
        # Direct translation ignoring start/stop
        df["Protein"] = df["RNA"].apply(
            lambda s: translate_mrna(s, frame=0, stop_at_stop=False)
        )
    else:
        # Biological ORF mode; empty string if no ORF found
        df["Protein"] = df["RNA"].apply(
            lambda s: translate_longest_orf(s) or ""
        )

    # reorder columns
    cols = ["DNA", "Protein", "RNA"] + [
        c for c in df.columns if c not in {"DNA", "Protein", "RNA"}
    ]
    df = df[cols]

    out_path = path.with_name(f"{path.stem}_multimodal.csv")
    df.to_csv(out_path, index=False)
    print(f"  -> wrote {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Convert RNA CSVs to multimodal CSVs.")
    parser.add_argument("input_dir", type=str, help="Folder with .csv files.")
    parser.add_argument(
        "--ignore_start_stop",
        action="store_true",
        help="translate RNA directly without requiring start/stop codons"
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files in {input_dir}")
        return

    for csv_path in csv_files:
        process_csv_file(csv_path, ignore_start_stop=args.ignore_start_stop)


if __name__ == "__main__":
    main()
