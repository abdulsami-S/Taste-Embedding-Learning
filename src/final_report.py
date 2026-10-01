import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

ALPHA = 0.5   # weight on the raw embedding

df = pd.read_pickle("data/processed/recipes_10k.pkl")
raw = np.load("data/processed/embeddings_text_A.npy")
learned = np.load("data/processed/embeddings_taste.npy")
final = np.hstack([np.sqrt(ALPHA) * raw, np.sqrt(1 - ALPHA) * learned])
np.save("data/processed/embeddings_final.npy", final)

PRIORITY = ["dessert", "breakfast", "salad", "soup", "casserole", "side", "lunch", "dinner"]
def main_category(terms):
    for p in PRIORITY:
        if p in terms:
            return p
    return "other"
cats = df["search_terms"].apply(main_category).to_numpy(dtype=object)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["search_terms"])
Y = Y[:, Y.sum(axis=0) >= 50].astype(np.float32)

idx = np.arange(len(df))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

def heldout(vecs, k=10):
    sims = vecs[test_idx] @ vecs[train_idx].T
    top = np.argsort(-sims, axis=1)[:, :k]
    nb = train_idx[top]
    cat = (cats[nb] == cats[test_idx][:, None]).mean(1)
    yt, yn = Y[test_idx], Y[nb]
    inter = (yt[:, None, :] * yn).sum(2)
    union = yt.sum(1)[:, None] + yn.sum(2) - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1), 0).mean(1)
    return cat, jac

p = pd.Series(cats[train_idx]).value_counts(normalize=True)
chance = float((p ** 2).sum())

systems = {"raw (pretrained)": raw, "learned": learned, f"blend (raw weight {ALPHA})": final}
overall, per_cat = [], {}
for name, vecs in systems.items():
    cat, jac = heldout(vecs)
    overall.append({"model": name, "category_match@10": cat.mean(), "jaccard@10": jac.mean()})
    per_cat[name] = pd.Series(cat).groupby(cats[test_idx]).mean()

overall.append({"model": "random (chance)", "category_match@10": chance, "jaccard@10": np.nan})
overall = pd.DataFrame(overall).round(3)
per_cat = pd.DataFrame(per_cat).round(3)
print(overall.to_string(index=False))
print("\n", per_cat)
overall.to_csv("results/final_results.csv", index=False)
per_cat.to_csv("results/per_category_results.csv")

# ---- Figure 1: per-category comparison ----
ax = per_cat.plot.bar(figsize=(10, 5))
ax.axhline(chance, color="gray", linestyle="--", label="chance")
ax.set_ylabel("category match@10 (held-out recipes)")
ax.set_title("Neighbor category match by model")
ax.legend()
plt.tight_layout()
plt.savefig("results/per_category_comparison.png", dpi=200)
plt.close()

# ---- Figure 2: map of the final embedding space ----
coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(final)
plt.figure(figsize=(9, 8))
for c in PRIORITY + ["other"]:
    m = cats == c
    plt.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.6, label=c)
plt.legend(markerscale=4)
plt.title("Final taste embedding (blend)")
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.savefig("results/final_embedding_map.png", dpi=200)
print("saved results figures")