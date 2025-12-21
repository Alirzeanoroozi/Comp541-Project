import pandas as pd
from torch.utils.data import DataLoader
from data.loaders.unimodal_dataset import UnimodalDataset

def get_loaders(name, batch_size=32, modality="RNA"): 
    df = pd.read_csv(f"data/datasets/{name}_multimodal.csv")

    df["Value"] = df["Value"].astype(int)
    train_df = df[df["Split"] == "train"]
    val_df = df[df["Split"] == "val"]
    test_df = df[df["Split"] == "test"]

    embedding_folder = f"embeddings/{name}/{modality}"

    train_dataset = UnimodalDataset(embedding_folder=embedding_folder, sequences=train_df[modality].tolist(), labels=train_df["Value"].tolist(), ids=train_df["id"].tolist())
    val_dataset = UnimodalDataset(embedding_folder=embedding_folder, sequences=val_df[modality].tolist(), labels=val_df["Value"].tolist(), ids=val_df["id"].tolist())
    test_dataset = UnimodalDataset(embedding_folder=embedding_folder, sequences=test_df[modality].tolist(), labels=test_df["Value"].tolist(), ids=test_df["id"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
