import pandas as pd
import requests
import os
from io import BytesIO

urls = [
    "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/E.Coli_proteins.csv",
    "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/Fungal_expression.csv",
    "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/mRNA_Stability.csv",
    "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/refs/heads/master/benchmarks/CodonBERT/data/fine-tune/CoV_Vaccine_Degradation.csv"
]
names = [
    "ecoli_proteins",
    "fungal_expression",
    "mrna_stability",
    "cov_vaccine_degradation"
]

os.makedirs("datasets", exist_ok=True)

def download_data(url):
    response = requests.get(url)
    return pd.read_csv(BytesIO(response.content))

def main():
    for url in urls:
        data = download_data(url)
        data.to_csv(f"datasets/{names[urls.index(url)]}.csv", index=False)

if __name__ == "__main__":
    main()