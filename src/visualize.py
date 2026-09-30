import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

df = pd.read_pickle("data/processed/recipes_10k.pkl")

# search_terms is multi-label, so pick one main category per recipe for coloring
PRIORITY = ["dessert", "breakfast", "salad", "soup", "casserole", "side", "lunch", "dinner"]

def main_category(terms):
    for p in PRIORITY:
        if p in terms:
            return p
    return "other"

df["category"] = df["search_terms"].apply(main_category)
print(df["category"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, version in zip(axes, ["text_A", "text_B"]):
    vectors = np.load(f"data/processed/embeddings_{version}.npy")
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                       random_state=42).fit_transform(vectors)

    for cat in PRIORITY + ["other"]:
        mask = (df["category"] == cat).values
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.6, label=cat)

    ax.set_title(f"Recipe embeddings ({version})")
    ax.set_xticks([]); ax.set_yticks([])

axes[0].legend(markerscale=4, loc="best")
plt.tight_layout()
plt.savefig("results/embedding_map.png", dpi=200)
print("saved results/embedding_map.png")