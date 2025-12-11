import pandas as pd
from torch.utils.data import DataLoader
from data.loaders.rna_dataset import RNADataset

def get_loaders(name, batch_size=32): 
    df = pd.read_csv(f"data/datasets/{name}.csv")

    df["Value"] = df["Value"].astype(int)
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
