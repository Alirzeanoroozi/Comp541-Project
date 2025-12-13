# Installation
conda env create --file environment.yml
----

(i) codon-level embedding concatenation,
(ii) entropy-regularized attention pooling inspired by multiple-instance learning
(iii) crossmodal multi-head attention—

Methods
the Nucleotide Transformer (Dalla-Torre et al., 2023) for DNA (6-mer tokenization),
RNA-FM (Chen et al., 2022) for RNA (single nucleotide tokenization),
ESM-2 (Lin et al., 2022a) for protein (amino acid tokenization, i.e., 3-mers in nucleotide).

| **Activity**                                   | **Deadline**  |
|------------------------------------------------|---------------|
| Gathering Data and implement BioLang           | 11/30/2025    |
| Train the model                                | 12/07/2025    |
| Project Report                                 | 12/21/2025    |
| Add other modality and fine-tuning code        | 01/04/2026    |
| Train the other modality and fine-tuning       | 01/11/2026    |
| Prepare final report and presentation          | 01/25/2026    |

**Table:** Timeline of major project activities



Comp541-Project/
│
├── README.md
├── environment.yml
├── setup.py
│
├── configs/
│   ├── dna_config.yaml
│   ├── rna_config.yaml
│   ├── protein_config.yaml
│   ├── fusion_concat.yaml
│   ├── fusion_mil.yaml
│   ├── fusion_xattn.yaml
│   ├── lora_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── cov_vac/
│   │   ├── fungal/
│   │   ├── ecoli/
│   │   ├── mrna_stability/
│   │   ├── ab1/
│   ├── processed/
│   ├── loaders/
│       ├── dna_dataset.py
│       ├── rna_dataset.py
│       ├── protein_dataset.py
│       ├── multimodal_dataset.py
│
├── models/
│   ├── __init__.py
│   ├── dna_model.py              ← Nucleotide Transformer
│   ├── rna_model.py              ← RNA-FM (or ESM1b)
│   ├── protein_model.py          ← ESM-2
│   ├── text_model.py             ← BioBERT/SciBERT
│   ├── projection_heads.py       ← linear projection to shared latent
│   ├── fusion_concat.py
│   ├── fusion_mil.py
│   ├── fusion_xattn.py
│   ├── prediction_head.py        ← TextCNN or MLP
│   ├── lora_adapter.py           ← Low-Rank Adaptation
│
├── training/
│   ├── train_concat.py
│   ├── train_mil.py
│   ├── train_xattn.py
│   ├── train_lora.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── metrics.py
│
├── evaluation/
│   ├── eval_regression.py
│   ├── eval_classification.py
│   ├── eval_multimodal_transfer.py
│
├── analysis/
│   ├── visualize_embeddings.py
│   ├── tSNE_plots.ipynb
│   ├── compare_modalities.ipynb
│   ├── cross_task_transfer.ipynb
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_concat.ipynb
│   ├── 03_train_mil.ipynb
│   ├── 04_train_xattn.ipynb
│   ├── 05_lora_finetuning.ipynb
│   ├── 06_transfer_learning.ipynb
│   ├── 07_results_summary.ipynb
│
└── results/
    ├── trained_models/
    ├── predictions/
    ├── plots/
    └── logs/
