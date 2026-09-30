import numpy as np
import pandas as pd

df = pd.read_pickle("data/processed/recipes_10k.pkl")

PRIORITY = ["dessert", "breakfast", "salad", "soup", "casserole", "side", "lunch", "dinner"]

def main_category(terms):
    for p in PRIORITY:
        if p in terms:
            return p
    return "other"

df["category"] = df["search_terms"].apply(main_category)
labels = df["category"].values
term_sets = df["search_terms"].apply(set).tolist()

def evaluate(vectors, k=10, chunk=1000):
    n = len(vectors)
    same_cat = np.zeros(n)
    jaccard = np.zeros(n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = vectors[start:end] @ vectors.T
        for i in range(end - start):
            sims[i, start + i] = -np.inf          # exclude the recipe itself
        top = np.argsort(-sims, axis=1)[:, :k]
        for i in range(end - start):
            gi = start + i
            nbrs = top[i]
            same_cat[gi] = np.mean(labels[nbrs] == labels[gi])
            a = term_sets[gi]
            js = []
            for j in nbrs:
                b = term_sets[j]
                js.append(len(a & b) / len(a | b) if (a | b) else 0)
            jaccard[gi] = np.mean(js)
    return same_cat, jaccard

# Chance level: probability two random recipes share a category
p = pd.Series(labels).value_counts(normalize=True)
chance = float((p ** 2).sum())

rows = []
for version in ["text_A", "text_B"]:
    vectors = np.load(f"data/processed/embeddings_{version}.npy")
    same_cat, jaccard = evaluate(vectors, k=10)
    rows.append({"version": version,
                 "category_match@10": same_cat.mean(),
                 "label_overlap@10 (Jaccard)": jaccard.mean()})
    print(f"\n{version}: per-category match@10")
    print(pd.Series(same_cat).groupby(labels).mean().round(3).sort_values(ascending=False))

results = pd.DataFrame(rows)
results.loc[len(results)] = ["random (chance)", chance, np.nan]
print("\n", results.round(3))
results.to_csv("results/knn_evaluation.csv", index=False)