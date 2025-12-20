import os
import torch
import pandas as pd
from tqdm import tqdm
from models.rna_model import RNAFMEmbedder
from models.protein_model import ESM2Embedder
from models.dna_model import NucleotideTransformerEmbedder

def read_sequences_from_file(file_path, modality):
    df = pd.read_csv(file_path)
    seqs = df[modality].tolist()
    return seqs

def save_embedding(embedding, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embedding.cpu(), save_path)

def process_dataset_embedding(dataset, modality, model):
    sequences = read_sequences_from_file(f"data/datasets/{dataset}_multimodal.csv", modality)
    out_folder = os.path.join("embeddings", dataset, modality)
    os.makedirs(out_folder, exist_ok=True)
    for idx, seq in enumerate(tqdm(sequences, desc=f"Embedding {dataset} from {modality}")):
        emb = model(seq)
        save_fp = os.path.join(out_folder, f"seq{idx+1}.pt")
        save_embedding(emb, save_fp)

def calculate_embeddings(dataset, modality, device):
    print("Calculating embeddings... for dataset: ", dataset, "and modality: ", modality)
    print("=" * 60)

    if modality == 'RNA':
        embedder = RNAFMEmbedder(device=device)
    elif modality == 'Protein':
        embedder = ESM2Embedder(device=device)
    elif modality == 'DNA':
        embedder = NucleotideTransformerEmbedder(device=device)
    else:
        raise ValueError(f"Invalid modality: {modality}")
    
    process_dataset_embedding(dataset, modality, embedder)
