"""
01_eda_sephora.py
Initial exploratory profiling of the Sephora Products & Skincare Reviews dataset.
Outputs a markdown report to reports/01_sephora_eda.md and prints a summary.

Reads reviews in chunks to stay memory-safe on ~1M rows.
"""
import os
import glob
import pandas as pd
import numpy as np

DATA = os.path.join("data", "sephora", "archive")
OUT = os.path.join("reports", "01_sephora_eda.md")
lines = []


def w(s=""):
    print(s)
    lines.append(s)


def pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


# ---------------------------------------------------------------- PRODUCTS
w("# Sephora Dataset — Initial EDA\n")
prod = pd.read_csv(os.path.join(DATA, "product_info.csv"))
w("## 1. Products (`product_info.csv`)\n")
w(f"- Rows (products): **{len(prod):,}**")
w(f"- Columns: **{prod.shape[1]}**")
w(f"- Unique brands: **{prod['brand_name'].nunique():,}**")

# missingness for key columns
key_cols = ["rating", "reviews", "ingredients", "price_usd", "primary_category",
            "secondary_category", "tertiary_category", "highlights", "size"]
w("\n**Missingness (key product columns):**\n")
w("| column | missing | % |")
w("|---|---|---|")
for c in key_cols:
    if c in prod.columns:
        m = prod[c].isna().sum()
        w(f"| {c} | {m:,} | {pct(m, len(prod))} |")

# price + rating distributions
w("\n**Price (USD):**")
pr = prod["price_usd"].describe(percentiles=[.25, .5, .75, .95])
w(f"- min {pr['min']:.2f} | median {pr['50%']:.2f} | mean {pr['mean']:.2f} | "
  f"95th {pr['95%']:.2f} | max {pr['max']:.2f}")

w("\n**Product rating (0-5):**")
rr = prod["rating"].describe()
w(f"- mean {rr['mean']:.2f} | median {prod['rating'].median():.2f} | "
  f"min {rr['min']:.2f} | max {rr['max']:.2f}")

w("\n**Top primary categories:**\n")
w("| category | products |")
w("|---|---|")
for cat, n in prod["primary_category"].value_counts().head(12).items():
    w(f"| {cat} | {n:,} |")

# skincare focus
skin_mask = prod["primary_category"].astype(str).str.contains("Skincare", case=False, na=False)
w(f"\n- Products in a **Skincare** primary category: **{skin_mask.sum():,}** "
  f"({pct(skin_mask.sum(), len(prod))})")

# ---------------------------------------------------------------- REVIEWS
w("\n## 2. Reviews (`reviews_*.csv`)\n")
review_files = sorted(glob.glob(os.path.join(DATA, "reviews_*.csv")))
w(f"- Review files: **{len(review_files)}**")

total = 0
rating_counts = pd.Series(dtype="int64")
skin_type_counts = pd.Series(dtype="int64")
skin_tone_counts = pd.Series(dtype="int64")
is_rec_counts = pd.Series(dtype="int64")
missing_text = 0
missing_skin_type = 0
missing_skin_tone = 0
text_len_sum = 0
text_len_n = 0
empty_text = 0
unique_products = set()
unique_authors = set()

usecols = ["author_id", "rating", "is_recommended", "review_text",
           "skin_tone", "skin_type", "product_id"]

for f in review_files:
    for chunk in pd.read_csv(f, usecols=usecols, chunksize=100_000,
                             low_memory=False):
        n = len(chunk)
        total += n
        rating_counts = rating_counts.add(
            chunk["rating"].value_counts(), fill_value=0)
        skin_type_counts = skin_type_counts.add(
            chunk["skin_type"].value_counts(), fill_value=0)
        skin_tone_counts = skin_tone_counts.add(
            chunk["skin_tone"].value_counts(), fill_value=0)
        is_rec_counts = is_rec_counts.add(
            chunk["is_recommended"].value_counts(dropna=False)
            .rename(lambda x: str(x)), fill_value=0)
        missing_text += chunk["review_text"].isna().sum()
        missing_skin_type += chunk["skin_type"].isna().sum()
        missing_skin_tone += chunk["skin_tone"].isna().sum()
        txt = chunk["review_text"].dropna().astype(str)
        text_len_sum += txt.str.len().sum()
        text_len_n += len(txt)
        empty_text += (txt.str.strip() == "").sum()
        unique_products.update(chunk["product_id"].dropna().unique())
        unique_authors.update(chunk["author_id"].dropna().unique())

w(f"- Total reviews: **{total:,}**")
w(f"- Unique products reviewed: **{len(unique_products):,}**")
w(f"- Unique reviewers (author_id): **{len(unique_authors):,}**")
w(f"- Missing review_text: **{missing_text:,}** ({pct(missing_text, total)})")
w(f"- Avg review length: **{(text_len_sum / text_len_n):.0f}** chars")

w("\n**Rating distribution (reviews):**\n")
w("| stars | count | % |")
w("|---|---|---|")
for stars in sorted(rating_counts.index):
    c = int(rating_counts[stars])
    w(f"| {stars} | {c:,} | {pct(c, total)} |")

w("\n**Reviewer skin_type (self-reported):**\n")
w("| skin_type | count | % |")
w("|---|---|---|")
for k, v in skin_type_counts.sort_values(ascending=False).items():
    w(f"| {k} | {int(v):,} | {pct(int(v), total)} |")
w(f"| (missing) | {missing_skin_type:,} | {pct(missing_skin_type, total)} |")

w("\n**Reviewer skin_tone (self-reported) — key for fairness:**\n")
w("| skin_tone | count | % |")
w("|---|---|---|")
for k, v in skin_tone_counts.sort_values(ascending=False).items():
    w(f"| {k} | {int(v):,} | {pct(int(v), total)} |")
w(f"| (missing) | {missing_skin_tone:,} | {pct(missing_skin_tone, total)} |")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
w(f"\n\nReport written to {OUT}")
