import pandas as pd
from torch.utils.data import DataLoader
from loaders.rna_dataset import RNADataset

def get_ecoli_loaders(batch_size=32):
    url = "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/E.Coli_proteins.csv"
    df = pd.read_csv(url)

    df["Value"] = df["Value"].astype(int)
    ecoli_df = df[df["Dataset"] == "E.Coli proteins"]

    train_df = ecoli_df[ecoli_df["Split"] == "train"]
    val_df = ecoli_df[ecoli_df["Split"] == "val"]
    test_df = ecoli_df[ecoli_df["Split"] == "test"]

    train_dataset = RNADataset(sequences=train_df["Sequence"].tolist(), labels=train_df["Value"].tolist())
    val_dataset = RNADataset(sequences=val_df["Sequence"].tolist(), labels=val_df["Value"].tolist())
    test_dataset = RNADataset(sequences=test_df["Sequence"].tolist(), labels=test_df["Value"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
