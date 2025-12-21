import os
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_embeddings(dataset, modality):
    emb_dir = os.path.join("embeddings", dataset, modality)
    # sorted by seqX.pt filename
    files = sorted([f for f in os.listdir(emb_dir) if f.endswith('.pt')],
                   key=lambda x: int(x.replace('seq', '').replace('.pt', '')))
    embeddings = []
    for f in files:
        emb = torch.load(os.path.join(emb_dir, f), map_location=torch.device('cpu'))
        # flatten as needed (e.g., take mean over sequence dim if shape > 1D)
        if emb.dim() > 1:
            emb = emb.mean(dim=0)
        embeddings.append(emb.numpy())
    return np.stack(embeddings)

def load_values(dataset, modality):
    df = pd.read_csv(f"data/datasets/{dataset}_multimodal.csv")
    if 'Value' not in df.columns:
        raise ValueError("'Value' column not found in the dataset CSV.")
    return df['Value'].to_numpy()

def main(dataset, modality, perplexity=30, random_state=42):
    print(f"Loading embeddings for dataset='{dataset}', modality='{modality}'")
    X = load_embeddings(dataset, modality)
    print(f"Embeddings shape: {X.shape}")

    print("Loading values for coloring...")
    y = load_values(dataset, modality)
    if len(y) != X.shape[0]:
        raise ValueError("Number of values does not match number of embeddings.")

    print(f"Running t-SNE with perplexity={perplexity}...")
    X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(X)

    print("Plotting...")
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis', s=40, alpha=0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Value')
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    plt.title(f'TSNE plot for {dataset} ({modality}) colored by Value')
    plt.tight_layout()

    plt.savefig(f"TSNE_plot/{dataset}_{modality}.png")
    print(f"Plot saved to TSNE_plot/{dataset}_{modality}.png")

if __name__ == "__main__":
    dataset = "cov_vaccine_degradation"
    modality = "RNA"
    main(dataset, modality)
