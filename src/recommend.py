import sys
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from projector import Projector

ALPHA = 0.5
df = pd.read_pickle("data/processed/recipes_10k.pkl")
final = np.load("data/processed/embeddings_final.npy")

base = SentenceTransformer("all-MiniLM-L6-v2")
proj = Projector()
proj.load_state_dict(torch.load("results/taste_projector.pt"))
proj.eval()

def embed(name, ingredients):
    text = f"{name}. Ingredients: {', '.join(ingredients)}."
    v = base.encode(text, normalize_embeddings=True)
    with torch.no_grad():
        z = proj(torch.tensor(v, dtype=torch.float32).unsqueeze(0)).numpy()[0]
    return np.hstack([np.sqrt(ALPHA) * v, np.sqrt(1 - ALPHA) * z])

if __name__ == "__main__":
    name = sys.argv[1]
    ingredients = [x.strip() for x in sys.argv[2].split(",")]
    q = embed(name, ingredients)
    scores = final @ q
    for i in np.argsort(-scores)[:5]:
        print(f"{scores[i]:.3f}  {df['name'].iloc[i]}")