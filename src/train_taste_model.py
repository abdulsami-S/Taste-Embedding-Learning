import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from projector import Projector

torch.manual_seed(42)
np.random.seed(42)

# ---------- Load data ----------
df = pd.read_pickle("data/processed/recipes_10k.pkl")
X = np.load("data/processed/embeddings_text_A.npy").astype(np.float32)

PRIORITY = ["dessert", "breakfast", "salad", "soup", "casserole", "side", "lunch", "dinner"]
def main_category(terms):
    for p in PRIORITY:
        if p in terms:
            return p
    return "other"

cats = df["search_terms"].apply(main_category).values

# Multi-hot label matrix; keep only search terms that appear at least 50 times
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["search_terms"])
keep = Y.sum(axis=0) >= 50
Y = Y[:, keep].astype(np.float32)
print("Number of label terms used:", Y.shape[1])

# ---------- Train / test split ----------
idx = np.arange(len(df))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

Xtr, Xte = torch.tensor(X[train_idx]), torch.tensor(X[test_idx])
Ytr = torch.tensor(Y[train_idx])

model = Projector()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ---------- Loss: soft contrastive loss weighted by label overlap ----------
def soft_contrastive_loss(z, y, temp=0.1):
    inter = y @ y.T
    union = y.sum(1, keepdim=True) + y.sum(1, keepdim=True).T - inter
    jac = inter / union.clamp(min=1)                       # label overlap in [0, 1]

    eye = torch.eye(len(z), dtype=torch.bool)
    sim = (z @ z.T / temp).masked_fill(eye, -1e9)
    target = jac.masked_fill(eye, 0)
    target = target / target.sum(1, keepdim=True).clamp(min=1e-8)

    log_prob = F.log_softmax(sim, dim=1)
    return -(target * log_prob).sum(1).mean()

# ---------- Evaluation: test recipes retrieve neighbors from the training set ----------
def evaluate(emb_train, emb_test, k=10):
    sims = emb_test @ emb_train.T
    top = np.argsort(-sims, axis=1)[:, :k]
    match, jac = [], []
    for row, nbrs in enumerate(top):
        gi = test_idx[row]
        nb = train_idx[nbrs]
        match.append(np.mean(cats[nb] == cats[gi]))
        a = Y[gi] > 0
        js = []
        for j in nb:
            b = Y[j] > 0
            u = (a | b).sum()
            js.append((a & b).sum() / u if u else 0)
        jac.append(np.mean(js))
    return float(np.mean(match)), float(np.mean(jac))

base = evaluate(X[train_idx], X[test_idx])
print(f"Baseline (raw embeddings): category_match@10={base[0]:.3f}  jaccard@10={base[1]:.3f}")

# ---------- Training ----------
EPOCHS, BATCH = 30, 256
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr))
    total = 0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]
        if len(b) < 8:
            continue
        loss = soft_contrastive_loss(model(Xtr[b]), Ytr[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            ztr = model(Xtr).numpy()
            zte = model(Xte).numpy()
        m = evaluate(ztr, zte)
        print(f"epoch {epoch:2d}  loss={total:.3f}  category_match@10={m[0]:.3f}  jaccard@10={m[1]:.3f}")

# ---------- Save ----------
model.eval()
with torch.no_grad():
    all_z = model(torch.tensor(X)).numpy()
np.save("data/processed/embeddings_taste.npy", all_z)
torch.save(model.state_dict(), "results/taste_projector.pt")
print("saved embeddings_taste.npy and taste_projector.pt")