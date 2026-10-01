import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from projector import Projector

df = pd.read_pickle("data/processed/recipes_10k.pkl")
raw = np.load("data/processed/embeddings_text_A.npy")
learned = np.load("data/processed/embeddings_taste.npy")

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

# Same split as training, so test recipes were never seen by the projector
idx = np.arange(len(df))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

def blend(r, z, a):
    return np.hstack([np.sqrt(a) * r, np.sqrt(1 - a) * z])

def heldout_eval(vecs, k=10):
    sims = vecs[test_idx] @ vecs[train_idx].T
    top = np.argsort(-sims, axis=1)[:, :k]
    nb = train_idx[top]                                    # (n_test, k)
    cat_match = (cats[nb] == cats[test_idx][:, None]).mean()
    yt = Y[test_idx]                                       # (n_test, L)
    yn = Y[nb]                                             # (n_test, k, L)
    inter = (yt[:, None, :] * yn).sum(2)
    union = yt.sum(1)[:, None] + yn.sum(2) - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1), 0).mean()
    return cat_match, jac

# ---- Custom dishes (not in the dataset) ----
base = SentenceTransformer("all-MiniLM-L6-v2")
proj = Projector()
proj.load_state_dict(torch.load("results/taste_projector.pt"))
proj.eval()

def embed_new(text):
    v = base.encode(text, normalize_embeddings=True)
    with torch.no_grad():
        z = proj(torch.tensor(v, dtype=torch.float32).unsqueeze(0)).numpy()[0]
    return v, z

dosa = embed_new("Dosa. Ingredients: rice, urad dal, fenugreek seeds, salt, oil.")
cb = embed_new("Chicken burger. Ingredients: chicken patty, burger bun, lettuce, tomato, mayonnaise, onion.")

def find(name_part):
    return df["name"].str.lower().str.contains(name_part.lower()).idxmax()
ib, ip, ic = find("burger"), find("pizza"), find("chocolate cake")

rows = []
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:          # a = weight on the raw embedding
    vecs = blend(raw, learned, a)
    cm, jc = heldout_eval(vecs)
    sim = lambda x, y: float(x @ y)
    b = vecs[ib]
    s_cb = sim(b, blend(*cb, a))
    s_pz = sim(b, vecs[ip])
    s_ds = sim(b, blend(*dosa, a))
    s_ck = sim(b, vecs[ic])
    rows.append({
        "raw_weight": a, "category_match@10": cm, "jaccard@10": jc,
        "burger~chicken_burger": s_cb, "burger~pizza": s_pz,
        "burger~dosa": s_ds, "burger~cake": s_ck,
        "order_ok": (s_cb > s_ds) and (s_pz > s_ds) and (s_ds > s_ck),
    })

out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
out.to_csv("results/blend_sweep.csv", index=False)