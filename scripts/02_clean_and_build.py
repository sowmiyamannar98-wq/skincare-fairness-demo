"""
02_clean_and_build.py
------------------------------------------------------------------
Cleans the Sephora data, builds the product-level recommender data
layer, and generates EDA charts.

Run:  python scripts/02_clean_and_build.py

Outputs:
  data/processed/products_clean.parquet        cleaned skincare products
  data/processed/product_review_agg.parquet    per-product review stats
  data/processed/recommender_products.parquet   merged layer (chatbot uses this)
  data/processed/product_skintone_matrix.parquet  fairness: reviews x skin_tone
  reports/figures/*.png                         charts
  reports/02_build_summary.md                   short summary
------------------------------------------------------------------
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # no display needed
import matplotlib.pyplot as plt

RAW = os.path.join("data", "sephora", "archive")
PROC = os.path.join("data", "processed")
FIG = os.path.join("reports", "figures")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CHUNK = 100_000
SKIN_TYPES = ["combination", "dry", "normal", "oily"]
summary = []


def log(s=""):
    print(s)
    summary.append(s)


# ============================================================ 1. PRODUCTS
log("[1/4] Cleaning products ...")
prod = pd.read_csv(os.path.join(RAW, "product_info.csv"))

# focus the project on skincare
prod = prod[prod["primary_category"].astype(str).str.contains(
    "Skincare", case=False, na=False)].copy()

# numeric hygiene
for c in ["price_usd", "rating", "reviews", "loves_count"]:
    prod[c] = pd.to_numeric(prod[c], errors="coerce")

# a rough ingredient count (many rows store a comma-separated string)
prod["ingredient_count"] = (
    prod["ingredients"].astype(str)
    .where(prod["ingredients"].notna(), "")
    .apply(lambda s: 0 if s.strip() == "" else len([x for x in s.split(",") if x.strip()]))
)

keep = ["product_id", "product_name", "brand_name", "price_usd", "rating",
        "reviews", "loves_count", "ingredients", "ingredient_count",
        "highlights", "primary_category", "secondary_category",
        "tertiary_category", "size", "out_of_stock", "new"]
prod = prod[[c for c in keep if c in prod.columns]]
prod.to_parquet(os.path.join(PROC, "products_clean.parquet"), index=False)
log(f"      skincare products kept: {len(prod):,}")


# ============================================================ 2. REVIEWS AGG
log("[2/4] Aggregating reviews (chunked) ...")
base_parts, stype_parts, stone_parts = [], [], []
usecols = ["rating", "is_recommended", "review_text",
           "skin_tone", "skin_type", "product_id"]

for f in sorted(glob.glob(os.path.join(RAW, "reviews_*.csv"))):
    for ch in pd.read_csv(f, usecols=usecols, chunksize=CHUNK, low_memory=False):
        ch["rating"] = pd.to_numeric(ch["rating"], errors="coerce")
        ch["is_recommended"] = pd.to_numeric(ch["is_recommended"], errors="coerce")
        ch["is_positive"] = (ch["rating"] >= 4).astype(int)          # sentiment proxy
        ch["text_len"] = ch["review_text"].astype(str).str.len()

        g = ch.groupby("product_id").agg(
            n_reviews=("rating", "size"),
            rating_sum=("rating", "sum"),
            rating_cnt=("rating", "count"),
            rec_sum=("is_recommended", "sum"),
            rec_cnt=("is_recommended", "count"),
            pos_sum=("is_positive", "sum"),
            textlen_sum=("text_len", "sum"),
        )
        base_parts.append(g)

        # per skin_type counts (recommender fit signal)
        st = (ch.dropna(subset=["skin_type"])
                .groupby(["product_id", "skin_type"]).size()
                .rename("n").reset_index())
        stype_parts.append(st)

        # per skin_tone counts (fairness / representation)
        sn = (ch.dropna(subset=["skin_tone"])
                .groupby(["product_id", "skin_tone"]).size()
                .rename("n").reset_index())
        stone_parts.append(sn)

# combine base
base = pd.concat(base_parts).groupby(level=0).sum()
base["mean_rating"] = base["rating_sum"] / base["rating_cnt"]
base["pct_recommended"] = base["rec_sum"] / base["rec_cnt"]
base["pct_positive"] = base["pos_sum"] / base["rating_cnt"]
base["avg_text_len"] = base["textlen_sum"] / base["n_reviews"]
agg = base[["n_reviews", "mean_rating", "pct_recommended",
            "pct_positive", "avg_text_len"]].reset_index()
agg.to_parquet(os.path.join(PROC, "product_review_agg.parquet"), index=False)
log(f"      products with reviews: {len(agg):,}")

# per skin_type wide table
stype = (pd.concat(stype_parts).groupby(["product_id", "skin_type"])["n"].sum()
         .reset_index()
         .pivot(index="product_id", columns="skin_type", values="n")
         .fillna(0))
stype.columns = [f"reviews_{c}" for c in stype.columns]
stype = stype.reset_index()

# per skin_tone wide table (fairness matrix)
stone = (pd.concat(stone_parts).groupby(["product_id", "skin_tone"])["n"].sum()
         .reset_index()
         .pivot(index="product_id", columns="skin_tone", values="n")
         .fillna(0).astype(int))
stone.to_parquet(os.path.join(PROC, "product_skintone_matrix.parquet"))
log(f"      skin_tone matrix shape: {stone.shape}")


# ============================================================ 3. MERGE LAYER
log("[3/4] Building merged recommender layer ...")
rec = (prod.merge(agg, on="product_id", how="left")
           .merge(stype, on="product_id", how="left"))
for c in [c for c in rec.columns if c.startswith("reviews_")]:
    rec[c] = rec[c].fillna(0).astype(int)
rec.to_parquet(os.path.join(PROC, "recommender_products.parquet"), index=False)
log(f"      recommender layer: {rec.shape[0]:,} rows x {rec.shape[1]} cols")


# ============================================================ 4. CHARTS
log("[4/4] Writing charts ...")

# a) product price distribution (capped for readability)
plt.figure(figsize=(7, 4))
prod["price_usd"].clip(upper=300).plot(kind="hist", bins=40, color="#c98a9b")
plt.title("Skincare product price (USD, capped at 300)")
plt.xlabel("price_usd"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "price_distribution.png"), dpi=120); plt.close()

# b) skin_tone representation across all reviews (fairness headline)
tone_totals = stone.sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
tone_totals.plot(kind="bar", color="#8a6ea3")
plt.title("Review counts by reviewer skin_tone (representation bias)")
plt.ylabel("reviews"); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "skin_tone_representation.png"), dpi=120); plt.close()

# c) skin_type mix across reviews
type_totals = (stype.drop(columns="product_id").sum()
               .rename(lambda x: x.replace("reviews_", "")))
plt.figure(figsize=(6, 4))
type_totals.sort_values(ascending=False).plot(kind="bar", color="#6ea38a")
plt.title("Review counts by reviewer skin_type")
plt.ylabel("reviews"); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(os.path.join(FIG, "skin_type_mix.png"), dpi=120); plt.close()

# d) mean_rating vs n_reviews (credibility view for recommender)
m = agg[agg["n_reviews"] >= 5]
plt.figure(figsize=(7, 4))
plt.scatter(m["n_reviews"], m["mean_rating"], s=8, alpha=0.3, color="#c98a9b")
plt.xscale("log")
plt.title("Product mean rating vs #reviews")
plt.xlabel("n_reviews (log)"); plt.ylabel("mean_rating"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "rating_vs_volume.png"), dpi=120); plt.close()

log("      charts -> reports/figures/*.png")

# fairness quick stat
deep_like = [c for c in stone.columns if c in
             ["deep", "rich", "dark", "ebony"]]
light_like = [c for c in stone.columns if c in
              ["fair", "fairLight", "light", "lightMedium", "porcelain"]]
d = int(stone[deep_like].sum().sum()); l = int(stone[light_like].sum().sum())
tot = int(stone.sum().sum())
log("")
log(f"Fairness snapshot: light-group reviews = {l:,} ({100*l/tot:.1f}%), "
    f"deep-group reviews = {d:,} ({100*d/tot:.1f}%)")

with open(os.path.join("reports", "02_build_summary.md"), "w",
          encoding="utf-8") as fh:
    fh.write("# Build summary\n\n" + "\n".join(f"- {s}" for s in summary if s))
log("\nDONE. Outputs in data/processed/ and reports/figures/")
