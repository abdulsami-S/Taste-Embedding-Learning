import ast
import pandas as pd

df = pd.read_csv("data/raw/recipes.csv")

# 1. Sample for speed
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# 2. Convert list/set-looking strings into real Python objects
def parse(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["ingredients"] = df["ingredients"].apply(parse)
df["tags"] = df["tags"].apply(parse)
df["search_terms"] = df["search_terms"].apply(parse)

# 3. Remove generic tags that carry no taste information
noise_tags = {
    "time-to-make", "course", "main-ingredient", "preparation",
    "occasion", "dietary", "equipment", "cuisine", "taste-mood",
}
df["tags"] = df["tags"].apply(lambda t: [x for x in t if x not in noise_tags])

# 4. Fill missing descriptions
df["description"] = df["description"].fillna("")

# 5. Build the text for embedding
# Version A: name + ingredients only (no labels, so we can evaluate fairly)
df["text_A"] = df.apply(
    lambda r: f"{r['name']}. Ingredients: {', '.join(r['ingredients'])}.", axis=1
)

# Version B: adds tags (taste/cuisine/diet hints)
df["text_B"] = df.apply(
    lambda r: f"{r['name']}. Ingredients: {', '.join(r['ingredients'])}. "
              f"Tags: {', '.join(r['tags'][:15])}.", axis=1
)

# 6. Save for the next step
df[["id", "name", "ingredients", "tags", "search_terms", "text_A", "text_B"]] \
    .to_pickle("data/processed/recipes_10k.pkl")

print(df.shape)
print(df["text_A"].iloc[0])
print(df["text_B"].iloc[0])