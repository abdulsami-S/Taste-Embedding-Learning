import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

df = pd.read_pickle("data/processed/recipes_10k.pkl")
model = SentenceTransformer(MODEL_NAME)

for version in ["text_A", "text_B"]:
    vectors = model.encode(
        df[version].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # makes cosine similarity a simple dot product
    )
    np.save(f"data/processed/embeddings_{version}.npy", vectors)
    print(version, vectors.shape)