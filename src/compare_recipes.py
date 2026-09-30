import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_pickle("data/processed/recipes_10k.pkl")
vectors = np.load("data/processed/embeddings_text_A.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return model.encode(text, normalize_embeddings=True)

def similarity(a, b):
    return float(np.dot(a, b))   # cosine similarity, since vectors are normalized

def find(name_part):
    """Return the first recipe whose name contains the given text."""
    idx = df["name"].str.lower().str.contains(name_part.lower()).idxmax()
    return idx

# Recipes not in the dataset, written in the same format as text_A
dosa = embed("Dosa. Ingredients: rice, urad dal, fenugreek seeds, salt, oil.")
chicken_burger = embed(
    "Chicken burger. Ingredients: chicken patty, burger bun, lettuce, tomato, mayonnaise, onion."
)

burger = vectors[find("burger")]
cake = vectors[find("chocolate cake")]
pizza = vectors[find("pizza")]

pairs = {
    "burger vs chicken burger": similarity(burger, chicken_burger),
    "burger vs pizza": similarity(burger, pizza),
    "burger vs cake": similarity(burger, cake),
    "burger vs dosa": similarity(burger, dosa),
}
for k, v in pairs.items():
    print(f"{k:28s} {v:.3f}")

# Top 5 most similar recipes to a given one
def top_k(query_vec, k=5):
    scores = vectors @ query_vec
    best = np.argsort(-scores)[:k]
    for i in best:
        print(f"{scores[i]:.3f}  {df['name'].iloc[i]}")

print("\nClosest to the burger recipe:")
top_k(burger)
print("\nClosest to dosa:")
top_k(dosa)