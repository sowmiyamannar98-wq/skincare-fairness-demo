"""
recommender.py
------------------------------------------------------------------
Content-based skincare recommender (proposal section 7.5).

Deterministic and inspectable by design. A learned collaborative-filtering
ranker is explicitly rejected: the data do not support it, and an opaque
ranker would obstruct the fairness analysis that is the project's purpose.
Every score decomposes into named components, so any recommendation can be
explained and audited.

Imported by:
  scripts/07_propagation_analysis.py   (RQ2)
  the conversational demonstrator      (section 7.7)

The language model in the demonstrator never calls anything here except
`recommend()`, and never reorders its output (section 7.7 separation).
------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd

PROC = os.path.join("data", "processed")
FLAGS = os.path.join(PROC, "product_ingredient_flags.parquet")

# ---- the mapping table, in executable form (documented in reports/05) -------
# weight > 0 pulls toward a class, < 0 pushes away.
SEVERITY_MAP = {
    0: {"salicylic_acid": 1.0, "niacinamide": 0.8, "aha": 0.6, "pha": 0.5,
        "clay_charcoal": 0.3, "benzoyl_peroxide": -0.2},
    1: {"salicylic_acid": 1.0, "benzoyl_peroxide": 0.9, "retinoid": 0.8,
        "azelaic_acid": 0.8, "niacinamide": 0.7, "aha": 0.4},
    2: {"benzoyl_peroxide": 1.0, "retinoid": 1.0, "azelaic_acid": 0.9,
        "niacinamide": 0.7, "salicylic_acid": 0.6, "sulfur": 0.4},
    3: {"benzoyl_peroxide": 1.0, "retinoid": 1.0, "azelaic_acid": 1.0,
        "niacinamide": 0.6, "sulfur": 0.5},
}
REFERRAL_GRADES = {2, 3}

ATTRIBUTE_MAP = {
    "dry":        {"ceramide": 0.8, "hyaluronic_acid": 0.7, "glycerin": 0.4,
                   "squalane": 0.6, "panthenol": 0.5,
                   "denatured_alcohol": -0.8, "aha": -0.3},
    "sensitive":  {"centella": 0.8, "colloidal_oatmeal": 0.7, "allantoin": 0.5,
                   "panthenol": 0.5, "fragrance": -1.0, "essential_oil": -0.9,
                   "denatured_alcohol": -0.9},
    "oily":       {"clay_charcoal": 0.7, "zinc": 0.6, "niacinamide": 0.6,
                   "salicylic_acid": 0.5},
    "combination": {"niacinamide": 0.5, "pha": 0.4, "hyaluronic_acid": 0.3},
    "normal":     {"niacinamide": 0.3, "hyaluronic_acid": 0.3},
    "pih":        {"niacinamide": 0.9, "azelaic_acid": 0.9, "vitamin_c": 0.8,
                   "tranexamic_acid": 0.7, "alpha_arbutin": 0.6,
                   "licorice_root": 0.4},
}

# score component weights (documented; tuned for interpretability not accuracy)
W_INGREDIENT = 1.0
W_SENTIMENT = 0.35
W_RATING = 0.25
W_SKINTYPE = 0.20
W_IRRITANCY = 0.30


class SkincareRecommender:
    """Content-based ranker over the Sephora skincare catalogue."""

    def __init__(self, catalogue=None, min_reviews=5,
                 categories=("Moisturizers", "Treatments", "Cleansers",
                             "Masks", "Sunscreen", "Eye Care")):
        if catalogue is None:
            catalogue = pd.read_parquet(FLAGS)
        df = catalogue.copy()
        # only products we can actually reason about
        df = df[df["has_ingredients"] == 1]
        df = df[df["secondary_category"].isin(categories)]
        df = df[df["n_reviews"].fillna(0) >= min_reviews]
        df = df.reset_index(drop=True)
        self.cat = df
        self.classes = [c for c in SEVERITY_MAP[1]] + \
                       [c for m in ATTRIBUTE_MAP.values() for c in m]
        self.classes = sorted(set(self.classes) & set(df.columns))
        # normalised popularity / quality signals, computed once
        self.cat["_sent"] = self.cat["pct_positive"].fillna(
            self.cat["pct_positive"].median())
        self.cat["_rate"] = (self.cat["mean_rating"].fillna(
            self.cat["mean_rating"].median()) / 5.0)
        lv = np.log1p(self.cat["loves_count"].fillna(0))
        self.cat["_pop"] = (lv - lv.min()) / max(lv.max() - lv.min(), 1e-9)

    # ---------------------------------------------------------------- scoring
    def _ingredient_score(self, grade, attributes):
        weights = dict(SEVERITY_MAP.get(int(grade), SEVERITY_MAP[1]))
        for a in attributes:
            for cls, wt in ATTRIBUTE_MAP.get(a, {}).items():
                weights[cls] = weights.get(cls, 0.0) + wt
        score = np.zeros(len(self.cat), dtype=float)
        max_pos = sum(w for w in weights.values() if w > 0) or 1.0
        for cls, wt in weights.items():
            if cls in self.cat.columns:
                score += wt * self.cat[cls].values
        return score / max_pos, weights

    def _skintype_score(self, skin_type):
        col = {"dry": "hl_dry", "oily": "hl_oily",
               "combination": "hl_combo", "normal": None}.get(skin_type)
        if col is None or col not in self.cat.columns:
            return np.zeros(len(self.cat))
        return self.cat[col].values.astype(float)

    def recommend(self, grade, skin_type=None, sensitive=False,
                  pih_concern=False, budget=None, k=10, return_scores=False):
        """Return the top-k ranked products for an assessed grade plus the
        conversationally-declared attributes. Deterministic."""
        attributes = []
        if skin_type:
            attributes.append(skin_type)
        if sensitive:
            attributes.append("sensitive")
        if pih_concern:
            attributes.append("pih")

        ing, weights = self._ingredient_score(grade, attributes)
        sent = self.cat["_sent"].values
        rate = self.cat["_rate"].values
        stype = self._skintype_score(skin_type)
        irritant_penalty = np.zeros(len(self.cat))
        if sensitive:
            irritant_penalty = self.cat["n_irritants"].values / 3.0

        score = (W_INGREDIENT * ing
                 + W_SENTIMENT * sent
                 + W_RATING * rate
                 + W_SKINTYPE * stype
                 - W_IRRITANCY * irritant_penalty)

        out = self.cat.copy()
        out["score"] = score
        out["_ing"] = ing
        if budget is not None:
            out = out[out["price_usd"] <= budget]
        out = out.sort_values("score", ascending=False).head(k)
        cols = ["product_id", "product_name", "brand_name", "price_usd",
                "mean_rating", "n_reviews", "secondary_category", "score"]
        if return_scores:
            cols += ["_ing"] + [c for c in weights if c in out.columns]
        return out[cols].reset_index(drop=True)

    def popularity_baseline(self, k=10, budget=None):
        """Trivial ranker for the baseline comparison (proposal section 9)."""
        out = self.cat.copy()
        out["score"] = out["_pop"]
        if budget is not None:
            out = out[out["price_usd"] <= budget]
        out = out.sort_values("score", ascending=False).head(k)
        return out[["product_id", "product_name", "brand_name", "price_usd",
                    "mean_rating", "n_reviews", "secondary_category",
                    "score"]].reset_index(drop=True)

    # ---------------------------------------------------------------- support
    def acceptable_ingredient_classes(self, grade):
        """Reference standard for relevance scoring: the classes the mapping
        table endorses for a grade (positive weights only)."""
        return {c for c, w in SEVERITY_MAP.get(int(grade), {}).items() if w > 0}

    def precision_at_k(self, recs, grade):
        """Share of recommended products carrying >=1 endorsed class for the
        TRUE grade. Evaluated against a literature-referenced reference set of
        acceptable classes rather than a single 'correct' product, because no
        such ground truth exists (proposal section 9)."""
        ok = self.acceptable_ingredient_classes(grade)
        ok = [c for c in ok if c in self.cat.columns]
        if not ok or len(recs) == 0:
            return np.nan
        sub = self.cat.set_index("product_id").loc[
            [p for p in recs["product_id"] if p in set(self.cat["product_id"])],
            ok]
        return float((sub.sum(axis=1) > 0).mean())

    def needs_referral(self, grade):
        return int(grade) in REFERRAL_GRADES


if __name__ == "__main__":
    rec = SkincareRecommender()
    print(f"catalogue after filtering: {len(rec.cat)} products\n")
    for g in (0, 1, 2, 3):
        r = rec.recommend(grade=g, skin_type="oily", k=5)
        print(f"--- grade {g} (referral={rec.needs_referral(g)}) "
              f"P@5={rec.precision_at_k(r, g):.2f} ---")
        print(r[["product_name", "brand_name", "price_usd", "score"]]
              .to_string(index=False), "\n")
