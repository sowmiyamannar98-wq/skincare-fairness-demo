"""
03_rq5_review_audit.py
------------------------------------------------------------------
RQ5 — Independent representational audit of the Sephora review corpus
with respect to reviewer-reported skin tone.

This arm is deliberately independent of the image pipeline (proposal
sections 7.6 / RQ5): a substantive fairness contribution that stands
even if the facial tone range proves too narrow for a powered audit.

What it measures, disaggregated by reviewer skin-tone band:
  1. Representation ....... review volume + share, with Wilson CIs on share
  2. Rating .............. mean star rating + bootstrapped CI; Kruskal-Wallis
  3. Sentiment proxy ..... P(is_recommended) and P(rating>=4) + bootstrap CI
  4. Price ............... price of products reviewed, and of highly-rated
                          products, by tone (median + bootstrapped CI)
  5. Concern coverage .... share of reviews mentioning post-inflammatory
                          hyperpigmentation / dark-spot concerns (which
                          disproportionately affect darker skin), by tone

Run:  python scripts/03_rq5_review_audit.py

Outputs:
  data/processed/rq5_review_records.parquet   compact per-review table
  data/processed/rq5_tone_summary.csv         headline table
  reports/figures/rq5_*.png                   charts
  reports/03_rq5_review_audit.md              written audit chapter
------------------------------------------------------------------
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RAW = os.path.join("data", "sephora", "archive")
PROC = os.path.join("data", "processed")
FIG = os.path.join("reports", "figures")
OUT = os.path.join("reports", "03_rq5_review_audit.md")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CHUNK = 100_000
B = 1000                      # bootstrap resamples
RNG = np.random.default_rng(42)
lines = []


def w(s=""):
    print(s)
    lines.append(s)


def pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


# ---- tone banding -------------------------------------------------------
# Self-reported skin_tone values are collapsed into three ordered bands so
# that per-group estimates are dense enough to be informative. The mapping
# is documented and reported as a design artefact.
TONE_BANDS = {
    "porcelain": "Light", "fair": "Light", "fairLight": "Light",
    "light": "Light", "lightMedium": "Light",
    "medium": "Medium", "mediumTan": "Medium", "tan": "Medium",
    "olive": "Medium",
    "deep": "Deep", "rich": "Deep", "dark": "Deep", "ebony": "Deep",
}
BAND_ORDER = ["Light", "Medium", "Deep"]

# ---- concern lexicon (post-inflammatory hyperpigmentation family) -------
PIH_PATTERN = re.compile(
    r"\b(?:hyperpigmentation|dark spot|dark spots|post[- ]?inflammatory|"
    r"pih|discolou?ration|acne scar|acne scars|acne mark|acne marks|"
    r"melasma|brown spot|brown spots|hyperpigment)\b",
    re.IGNORECASE,
)


# ============================================================ 1. CHUNK PASS
w("# RQ5 — Representational Audit of the Sephora Review Corpus\n")
w("_Independent of the image pipeline. Reviewer skin tone is self-reported; "
  "tones are collapsed into three ordered bands (Light / Medium / Deep). "
  "Reviews with missing or non-committal tone (`notSureST`, blank) are "
  "excluded from disaggregated statistics and counted separately._\n")

records = []
n_total = 0
n_missing_tone = 0
n_notsure = 0

usecols = ["rating", "is_recommended", "skin_tone", "price_usd", "review_text"]
for f in sorted(glob.glob(os.path.join(RAW, "reviews_*.csv"))):
    for ch in pd.read_csv(f, usecols=usecols, chunksize=CHUNK, low_memory=False):
        n_total += len(ch)
        n_missing_tone += ch["skin_tone"].isna().sum()
        n_notsure += (ch["skin_tone"] == "notSureST").sum()

        ch = ch.copy()
        ch["band"] = ch["skin_tone"].map(TONE_BANDS)
        ch = ch[ch["band"].notna()]
        if ch.empty:
            continue
        ch["rating"] = pd.to_numeric(ch["rating"], errors="coerce")
        ch["is_recommended"] = pd.to_numeric(ch["is_recommended"], errors="coerce")
        ch["price_usd"] = pd.to_numeric(ch["price_usd"], errors="coerce")
        ch["is_positive"] = (ch["rating"] >= 4).astype("float")
        txt = ch["review_text"].astype(str)
        ch["pih"] = txt.str.contains(PIH_PATTERN).astype("float")
        records.append(ch[["band", "rating", "is_recommended",
                           "is_positive", "price_usd", "pih"]])

rec = pd.concat(records, ignore_index=True)
rec["band"] = pd.Categorical(rec["band"], categories=BAND_ORDER, ordered=True)
rec.to_parquet(os.path.join(PROC, "rq5_review_records.parquet"), index=False)

n_banded = len(rec)
w(f"- Total reviews scanned: **{n_total:,}**")
w(f"- Reviews with a usable tone band: **{n_banded:,}** "
  f"({pct(n_banded, n_total)})")
w(f"- Missing skin_tone: **{n_missing_tone:,}** ({pct(n_missing_tone, n_total)})")
w(f"- `notSureST` (declined): **{n_notsure:,}** ({pct(n_notsure, n_total)})\n")


# ============================================================ 2. BOOTSTRAP
def boot_ci(values, stat=np.mean, b=B, alpha=0.05):
    """Percentile bootstrap CI for a 1-D array statistic."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    point = stat(values)
    idx = RNG.integers(0, n, size=(b, n))
    boots = stat(values[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, lo, hi)


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion (used for representation share)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, centre - half, centre + half)


# ============================================================ 3. HEADLINE TABLE
rows = []
for band in BAND_ORDER:
    g = rec[rec["band"] == band]
    n = len(g)
    share = wilson(n, n_banded)
    mr = boot_ci(g["rating"].values)
    rec_rate = boot_ci(g["is_recommended"].values)
    pos_rate = boot_ci(g["is_positive"].values)
    price_med = boot_ci(g["price_usd"].values, stat=np.median)
    pih_rate = boot_ci(g["pih"].values)
    rows.append(dict(
        band=band, n=n,
        share=share[0], share_lo=share[1], share_hi=share[2],
        mean_rating=mr[0], mr_lo=mr[1], mr_hi=mr[2],
        pct_recommended=rec_rate[0], rec_lo=rec_rate[1], rec_hi=rec_rate[2],
        pct_positive=pos_rate[0], pos_lo=pos_rate[1], pos_hi=pos_rate[2],
        median_price=price_med[0], price_lo=price_med[1], price_hi=price_med[2],
        pih_rate=pih_rate[0], pih_lo=pih_rate[1], pih_hi=pih_rate[2],
    ))
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(PROC, "rq5_tone_summary.csv"), index=False)

w("## 1. Representation (review volume by tone band)\n")
w("| band | reviews | share | 95% CI on share |")
w("|---|---|---|---|")
for r in rows:
    w(f"| {r['band']} | {r['n']:,} | {r['share']*100:.1f}% | "
      f"[{r['share_lo']*100:.1f}%, {r['share_hi']*100:.1f}%] |")
imb = rows[0]['n'] / max(rows[2]['n'], 1)
w(f"\n**Representation ratio (Light : Deep) = {imb:.1f} : 1.** "
  "The corpus is dominated by lighter-tone reviewers; every downstream "
  "per-tone estimate for the Deep band rests on a far thinner sample, which "
  "is why intervals are reported throughout.\n")

w("## 2. Rating, sentiment, and price by tone (bootstrapped 95% CIs)\n")
w("| band | mean rating | P(recommended) | P(rating>=4) | median price (USD) |")
w("|---|---|---|---|---|")
for r in rows:
    w(f"| {r['band']} | {r['mean_rating']:.3f} "
      f"[{r['mr_lo']:.3f}, {r['mr_hi']:.3f}] | "
      f"{r['pct_recommended']*100:.1f}% "
      f"[{r['rec_lo']*100:.1f}, {r['rec_hi']*100:.1f}] | "
      f"{r['pct_positive']*100:.1f}% "
      f"[{r['pos_lo']*100:.1f}, {r['pos_hi']*100:.1f}] | "
      f"${r['median_price']:.0f} "
      f"[${r['price_lo']:.0f}, ${r['price_hi']:.0f}] |")
w("")


# ============================================================ 4. SIG TESTS
w("## 3. Significance tests and effect sizes\n")

# Kruskal-Wallis on rating across the three bands
groups_rating = [rec[rec["band"] == b]["rating"].dropna().values
                 for b in BAND_ORDER]
H, p_kw = stats.kruskal(*groups_rating)
# epsilon-squared effect size for Kruskal-Wallis
N = sum(len(g) for g in groups_rating)
eps2 = (H - len(groups_rating) + 1) / (N - len(groups_rating))
w(f"- **Rating ~ tone (Kruskal-Wallis):** H={H:.1f}, p={p_kw:.2e}, "
  f"epsilon-squared={eps2:.4f} "
  f"({'negligible' if eps2 < 0.01 else 'small' if eps2 < 0.06 else 'moderate'} "
  "effect). A significant p on ~1M rows is expected; the effect size is the "
  "honest measure of practical magnitude.")

# Price across bands
groups_price = [rec[rec["band"] == b]["price_usd"].dropna().values
                for b in BAND_ORDER]
Hp, p_price = stats.kruskal(*groups_price)
eps2p = (Hp - len(groups_price) + 1) / (N - len(groups_price))
w(f"- **Price ~ tone (Kruskal-Wallis):** H={Hp:.1f}, p={p_price:.2e}, "
  f"epsilon-squared={eps2p:.4f}.")

# PIH coverage: chi-square band x mentions-PIH
ct = pd.crosstab(rec["band"], rec["pih"])
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
# Cramer's V
cv = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
w(f"- **PIH-concern mention ~ tone (chi-square):** chi2={chi2:.1f}, "
  f"p={p_chi:.2e}, Cramer's V={cv:.4f}.\n")

w("## 4. Concern coverage — post-inflammatory hyperpigmentation family\n")
w("Share of each band's reviews that mention hyperpigmentation / dark-spot / "
  "acne-scarring concerns. These concerns disproportionately affect darker "
  "skin, so their prevalence by tone speaks to whether the corpus surfaces "
  "the needs of darker-skinned users.\n")
w("| band | reviews mentioning PIH concerns | rate | 95% CI |")
w("|---|---|---|---|")
for r in rows:
    g = rec[rec["band"] == r['band']]
    k = int(g["pih"].sum())
    w(f"| {r['band']} | {k:,} | {r['pih_rate']*100:.2f}% | "
      f"[{r['pih_lo']*100:.2f}%, {r['pih_hi']*100:.2f}%] |")
w("")


# ============================================================ 5. CHARTS
def _band_colors():
    return {"Light": "#e6b8b0", "Medium": "#b07d56", "Deep": "#5c3a26"}


col = _band_colors()

# a) representation
plt.figure(figsize=(6, 4))
plt.bar([r['band'] for r in rows], [r['n'] for r in rows],
        color=[col[r['band']] for r in rows])
plt.yscale("log")
plt.title("RQ5 — review volume by reviewer tone band (log scale)")
plt.ylabel("reviews (log)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq5_representation.png"), dpi=120)
plt.close()

# b) mean rating with CI
plt.figure(figsize=(6, 4))
xs = np.arange(len(rows))
plt.errorbar(xs, [r['mean_rating'] for r in rows],
             yerr=[[r['mean_rating'] - r['mr_lo'] for r in rows],
                   [r['mr_hi'] - r['mean_rating'] for r in rows]],
             fmt="o", capsize=5, color="#8a6ea3")
plt.xticks(xs, [r['band'] for r in rows])
plt.title("RQ5 — mean star rating by tone (95% bootstrap CI)")
plt.ylabel("mean rating")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq5_rating_ci.png"), dpi=120)
plt.close()

# c) median price with CI
plt.figure(figsize=(6, 4))
plt.errorbar(xs, [r['median_price'] for r in rows],
             yerr=[[r['median_price'] - r['price_lo'] for r in rows],
                   [r['price_hi'] - r['median_price'] for r in rows]],
             fmt="s", capsize=5, color="#6ea38a")
plt.xticks(xs, [r['band'] for r in rows])
plt.title("RQ5 — median price of reviewed products by tone (95% CI)")
plt.ylabel("median price (USD)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq5_price_ci.png"), dpi=120)
plt.close()

# d) PIH coverage with CI
plt.figure(figsize=(6, 4))
plt.errorbar(xs, [r['pih_rate'] * 100 for r in rows],
             yerr=[[(r['pih_rate'] - r['pih_lo']) * 100 for r in rows],
                   [(r['pih_hi'] - r['pih_rate']) * 100 for r in rows]],
             fmt="D", capsize=5, color="#c98a9b")
plt.xticks(xs, [r['band'] for r in rows])
plt.title("RQ5 — PIH-concern mention rate by tone (95% CI)")
plt.ylabel("% of reviews")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq5_pih_coverage.png"), dpi=120)
plt.close()

w("## 5. Figures\n")
w("- `reports/figures/rq5_representation.png`")
w("- `reports/figures/rq5_rating_ci.png`")
w("- `reports/figures/rq5_price_ci.png`")
w("- `reports/figures/rq5_pih_coverage.png`\n")

w("## 6. Reading these results\n")
w("- **Representation is the dominant finding.** The Light:Deep volume ratio "
  f"is {imb:.0f}:1; darker-skinned reviewers are severely under-represented, "
  "so the platform's aggregate review signal is overwhelmingly a lighter-skin "
  "signal. This is a representational bias in the corpus itself, before any "
  "model is trained on it.")
w("- **Effect sizes, not p-values, carry the argument.** With ~1M reviews, "
  "almost any difference is 'significant'; epsilon-squared and Cramer's V are "
  "reported so that trivial differences are not oversold.")
w("- **Concern coverage** indicates whether darker-skin concerns (PIH) are "
  "even voiced in the corpus at rates matching their real-world prevalence; a "
  "low absolute Deep-band volume means even a higher per-review rate "
  "translates into few reviews the recommender can learn from.")
w("- **Independence.** None of this depends on the image model; it is a "
  "self-contained contribution answering RQ5.\n")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
w(f"Report written to {OUT}")
