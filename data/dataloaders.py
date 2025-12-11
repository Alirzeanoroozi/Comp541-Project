import pandas as pd
from torch.utils.data import DataLoader
from loaders.rna_dataset import RNADataset

# ecoli: "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/E.Coli_proteins.csv"
# fungal: "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/Fungal_expression.csv"
# mrna_stab: "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/mRNA_Stability.csv"
# cov_vac: "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/CoV_Vaccine_Degradation.csv"
def get_loaders(url, batch_size=32): 
    df = pd.read_csv(url)

    df["Value"] = df["Value"].astype(int)
    if "Dataset" in df.columns:
        df = df[df["Dataset"] == "E.Coli proteins"]

    train_df = df[df["Split"] == "train"]
    val_df = df[df["Split"] == "val"]
    test_df = df[df["Split"] == "test"]

    # labels; 
    train_dataset = RNADataset(sequences=train_df["Sequence"].tolist(), labels=train_df["Value"].tolist())
    val_dataset = RNADataset(sequences=val_df["Sequence"].tolist(), labels=val_df["Value"].tolist())
    test_dataset = RNADataset(sequences=test_df["Sequence"].tolist(), labels=test_df["Value"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
