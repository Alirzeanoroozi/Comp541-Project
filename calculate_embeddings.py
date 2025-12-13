import os
import torch
import pandas as pd
from tqdm import tqdm
from utils.load_config import load_config
from models.rna_model import RNAFMEmbedder
from models.protein_model import ESM2Embedder
from models.dna_model import NucleotideTransformerEmbedder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = [
    "fungal_expression",
    # Add more datasets as needed, e.g.:
    # "mrna_stability",
    # "ecoli_proteins",
    # "cov_vaccine_degradation"
]

MODALITIES = [
    'Protein',
    # 'RNA',
    # 'DNA'
]

OUT_EMB_ROOT = "embeddings"

def read_sequences_from_file(file_path, modality):
    df = pd.read_csv(file_path)
    seqs = df[modality].tolist()
    return seqs

def save_embedding(embedding, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embedding.cpu(), save_path)

def process_dataset_embedding(dataset, modality, output_folder, model, max_len):

    sequences = read_sequences_from_file(f"data/datasets/{dataset}_multimodal.csv", modality)

    for idx, seq in enumerate(tqdm(sequences, desc=f"Embedding {dataset} from {modality}")):
        emb = model(seq)
        save_fp = os.path.join(output_folder, f"{dataset}_seq{idx+1}.pt")
        save_embedding(emb, save_fp)

def main():
    for dataset in DATASETS:
        for modality in MODALITIES:
            if modality == 'RNA':
                max_len = 1000
                embedder = RNAFMEmbedder(max_len=max_len, device=DEVICE)
            elif modality == 'Protein':
                max_len = 350
                embedder = ESM2Embedder(max_len=max_len, device=DEVICE)
            elif modality == 'DNA':
                max_len = 512
                embedder = NucleotideTransformerEmbedder(max_len=max_len, device=DEVICE)
            else:
                raise ValueError(f"Invalid modality: {modality}")
            embedder.eval()
            out_folder = os.path.join(OUT_EMB_ROOT, dataset, modality)
            os.makedirs(out_folder, exist_ok=True)
            process_dataset_embedding(dataset, modality, out_folder, embedder, max_len)

if __name__ == "__main__":
    main()
