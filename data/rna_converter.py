import pandas as pd
from tqdm import tqdm
from pathlib import Path

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
    """Uppercase, keep A/C/G/U, convert T → U."""
    seq = seq.upper().replace("T", "U")
    return "".join(b for b in seq if b in {"A", "C", "G", "U"})

def mrna_to_dna(mrna: str) -> str:
    return mrna.replace("U", "T")

def translate_frame0(mrna: str) -> str:
    """
    Translate RNA in frame-0, ignoring start/stop codons.
    Stop codons are kept as '*'.
    """
    aa = []
    for i in range(0, len(mrna) - 2, 3):
        codon = mrna[i:i + 3]
        aa.append(GENETIC_CODE.get(codon, "X"))
    return "".join(aa)

def process_csv_file(path: Path) -> None:
    """
    Reads CSV with column 'Sequence'
    Writes <stem>_multimodal.csv with columns:
        DNA, Protein, RNA, <all other original columns>
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

    # Protein (always frame-0)
    df["Protein"] = df["RNA"].apply(translate_frame0)
    
    df['id'] = [f"seq{i+1}" for i in range(len(df))]

    # Reorder columns
    cols = ["DNA", "Protein", "RNA"] + [
        c for c in df.columns if c not in {"DNA", "Protein", "RNA"}
    ]
    df = df[cols]

    out_path = path.with_name(f"{path.stem}_multimodal.csv")
    df.to_csv(out_path, index=False)
    print(f"  -> wrote {out_path.name}")

def main():
    input_dir = Path("data/datasets")

    csv_files = sorted([f for f in input_dir.glob("*.csv") if "multimodal" not in f.name])

    for csv_path in tqdm(csv_files, desc="Processing CSV files"):
        process_csv_file(csv_path)

if __name__ == "__main__":
    main()
