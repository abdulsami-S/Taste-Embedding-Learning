import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from projector import Projector

df = pd.read_pickle("data/processed/recipes_10k.pkl")
raw = np.load("data/processed/embeddings_text_A.npy")
learned = np.load("data/processed/embeddings_taste.npy")

base = SentenceTransformer("all-MiniLM-L6-v2")
proj = Projector()
proj.load_state_dict(torch.load("results/taste_projector.pt"))
proj.eval()

def embed_new(text):
    """Return (raw vector, learned vector) for a recipe not in the dataset."""
    v = base.encode(text, normalize_embeddings=True)
    with torch.no_grad():
        z = proj(torch.tensor(v, dtype=torch.float32).unsqueeze(0)).numpy()[0]
    return v, z

def find(name_part):
    return df["name"].str.lower().str.contains(name_part.lower()).idxmax()

dosa_raw, dosa_z = embed_new("Dosa. Ingredients: rice, urad dal, fenugreek seeds, salt, oil.")
cb_raw, cb_z = embed_new(
    "Chicken burger. Ingredients: chicken patty, burger bun, lettuce, tomato, mayonnaise, onion.")

i_burger, i_pizza, i_cake = find("burger"), find("pizza"), find("chocolate cake")

rows = [
    ("burger vs chicken burger", raw[i_burger] @ cb_raw,   learned[i_burger] @ cb_z),
    ("burger vs pizza",          raw[i_burger] @ raw[i_pizza], learned[i_burger] @ learned[i_pizza]),
    ("burger vs dosa",           raw[i_burger] @ dosa_raw, learned[i_burger] @ dosa_z),
    ("burger vs cake",           raw[i_burger] @ raw[i_cake],  learned[i_burger] @ learned[i_cake]),
]
table = pd.DataFrame(rows, columns=["pair", "raw_similarity", "learned_similarity"]).round(3)
print(table.to_string(index=False))
table.to_csv("results/burger_dosa_comparison.csv", index=False)

def top_k(vectors, q, k=5):
    scores = vectors @ q
    for i in np.argsort(-scores)[:k]:
        print(f"  {scores[i]:.3f}  {df['name'].iloc[i]}")

print("\nNearest to dosa (raw):");        top_k(raw, dosa_raw)
print("\nNearest to dosa (learned):");    top_k(learned, dosa_z)
print("\nNearest to chicken burger (raw):");     top_k(raw, cb_raw)
print("\nNearest to chicken burger (learned):"); top_k(learned, cb_z)