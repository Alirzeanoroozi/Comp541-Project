import pandas as pd
from torch.utils.data import DataLoader
from loaders.rna_dataset import RNADataset

def get_ecoli_loaders(batch_size=32):
    url = "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/E.Coli_proteins.csv"
    df = pd.read_csv(url)

    df["Value"] = df["Value"].astype(int)
    if "Dataset" in df.columns:
        df = df[df["Dataset"] == "E.Coli proteins"]

    train_df = df[ecoli_df["Split"] == "train"]
    val_df = df[ecoli_df["Split"] == "val"]
    test_df = df[ecoli_df["Split"] == "test"]

    # labels; 
    # 0=low expression; 1=moderate expression; 2=high expression
    train_dataset = RNADataset(sequences=train_df["Sequence"].tolist(), labels=train_df["Value"].tolist())
    val_dataset = RNADataset(sequences=val_df["Sequence"].tolist(), labels=val_df["Value"].tolist())
    test_dataset = RNADataset(sequences=test_df["Sequence"].tolist(), labels=test_df["Value"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_fungal_loaders(batch_size=32):
    url = "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/Fungal_expression.csv"
    df = pd.read_csv(url)

    if "Dataset" in df.columns:
        df = df[df["Dataset"] == "Fungal expression"]

    df["Value"] = df["Value"].astype(float)

    train_df = df[df["Split"] == "train"]
    val_df = df[df["Split"] == "val"]
    test_df = df[df["Split"] == "test"]

    train_dataset = RNADataset(sequences=train_df["Sequence"].tolist(), labels=train_df["Value"].tolist())
    val_dataset = RNADataset(sequences=val_df["Sequence"].tolist(), labels=val_df["Value"].tolist())
    test_dataset = RNADataset(sequences=test_df["Sequence"].tolist(), labels=test_df["Value"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_mrna_stability_loaders(batch_size=32):
    url = "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/mRNA_Stability.csv"
    df = pd.read_csv(url)

    if "Dataset" in df.columns:
        df = df[df["Dataset"] == "mRNA Stability"]

    df["Value"] = df["Value"].astype(float)

    train_df = df[df["Split"] == "train"]
    val_df = df[df["Split"] == "val"]
    test_df = df[df["Split"] == "test"]

    train_dataset = RNADataset(sequences=train_df["Sequence"].tolist(), labels=train_df["Value"].tolist())
    val_dataset = RNADataset(sequences=val_df["Sequence"].tolist(), labels=val_df["Value"].tolist())
    test_dataset = RNADataset(sequences=test_df["Sequence"].tolist(), labels=test_df["Value"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_cov_vac_loaders(batch_size=32):
    url = "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/CoV_Vaccine_Degradation.csv"
    df = pd.read_csv(url)

    if "Dataset" in df.columns:
        df = df[df["Dataset"] == "CoV Vaccine Degradation"]

    df["Value"] = df["Value"].astype(float)

    train_df = df[df["Split"] == "train"]
    val_df = df[df["Split"] == "val"]
    test_df = df[df["Split"] == "test"]

    train_dataset = RNADataset(sequences=train_df["Sequence"].tolist(), labels=train_df["Value"].tolist())
    val_dataset = RNADataset(sequences=val_df["Sequence"].tolist(), labels=val_df["Value"].tolist())
    test_dataset = RNADataset(sequences=test_df["Sequence"].tolist(), labels=test_df["Value"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
