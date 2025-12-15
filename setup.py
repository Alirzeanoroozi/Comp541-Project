from setuptools import setup, find_packages

setup(
    name="Comp541-Project",
    version="0.1.0",
    description="Multimodal Biological Language Modeling Framework (DNA + RNA + Protein + Text)",
    author=["Alireza Noroozi","Sahand Hassanizorgabad","Mustafa Serhat Aydin"]
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "einops",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
        "sentencepiece",
        "accelerate",
        "peft"
    ],
    python_requires=">=3.8",
)
