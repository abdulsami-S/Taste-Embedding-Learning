import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.projector import Projector

ALPHA = 0.5
df = None
final = None
base = None
proj = None

def load_all():
    global df, final, base, proj
    df = pd.read_pickle("data/processed/recipes_10k.pkl")
    final = np.load("data/processed/embeddings_final.npy")
    base = SentenceTransformer("all-MiniLM-L6-v2")
    proj = Projector()
    proj.load_state_dict(torch.load("results/taste_projector.pt", map_location="cpu"))
    proj.eval()

def embed_text(text: str) -> np.ndarray:
    v = base.encode(text, normalize_embeddings=True)
    with torch.no_grad():
        z = proj(torch.tensor(v, dtype=torch.float32).unsqueeze(0)).numpy()[0]
    return np.hstack([np.sqrt(ALPHA) * v, np.sqrt(1 - ALPHA) * z])

def embed_recipe(name: str, ingredients: list[str]) -> np.ndarray:
    text = f"{name}. Ingredients: {', '.join(ingredients)}."
    return embed_text(text)

def top_k(query_vec: np.ndarray, k: int) -> list[dict]:
    k = max(1, min(k, 10))
    scores = final @ query_vec
    top_indices = np.argsort(-scores)[:k]
    
    results = []
    for i in top_indices:
        row = df.iloc[i]
        results.append({
            "id": int(row["id"]) if "id" in row else None,
            "name": str(row["name"]),
            "ingredients": list(row["ingredients"]),
            "search_terms": sorted(list(row["search_terms"])) if "search_terms" in row else [],
            "score": round(float(scores[i]), 3)
        })
    return results

def get_recipe_count() -> int:
    return len(df) if df is not None else 0
