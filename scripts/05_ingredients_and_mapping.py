"""
05_ingredients_and_mapping.py
------------------------------------------------------------------
Proposal sections 7.3 (ingredient parsing) and 7.4 (severity-and-attribute to
ingredient mapping) -- the interpretable seam between the image module and the
recommender.

Two deliberate design positions, both defended in the dissertation:

1. Ingredient parsing is RULE-BASED, not learned. The Sephora ingredient field
   is inconsistent free text; a documented deterministic pipeline is more
   tractable and far more defensible than an opaque model, and every match is
   inspectable.

2. The mapping table is a DESIGN ARTEFACT referenced to dermatological
   literature rather than marketing copy. It is presented and defended as such.
   Principal reference point: Zaenglein et al., "Guidelines of care for the
   management of acne vulgaris", J Am Acad Dermatol 2016;74(5):945-73, plus
   standard cosmetic-dermatology sources for barrier/soothing agents. The
   system is a COSMETIC tool: severe grades trigger a referral flag, never a
   prescription-strength claim.

Run:  python scripts/05_ingredients_and_mapping.py

Outputs:
  data/processed/product_ingredient_flags.parquet  product x ingredient-class
  reports/05_mapping_table.md                      the documented mapping
------------------------------------------------------------------
"""
import os
import re
import ast
import numpy as np
import pandas as pd

PROC = os.path.join("data", "processed")
OUT_MD = os.path.join("reports", "05_mapping_table.md")
IN = os.path.join(PROC, "recommender_products.parquet")
lines = []


def w(s=""):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))
    lines.append(s)


# ============================================================ 1. VOCABULARY
# Controlled vocabulary of ingredient CLASSES. Each class maps to regex
# patterns matched against the normalised ingredient string. Patterns are
# deliberately conservative: a false negative costs a missed candidate, a false
# positive puts an unsuitable product in front of a user.
INGREDIENT_CLASSES = {
    # --- acne actives ---
    "salicylic_acid":   [r"salicylic acid", r"\bbeta hydroxy\b", r"\bbha\b",
                         r"willow bark extract", r"salix alba.*bark"],
    "benzoyl_peroxide": [r"benzoyl peroxide"],
    "retinoid":         [r"retinol\b", r"retinal\b", r"retinaldehyde",
                         r"retinyl (palmitate|propionate|acetate)",
                         r"adapalene", r"tretinoin", r"hydroxypinacolone retinoate",
                         r"granactive retinoid"],
    "azelaic_acid":     [r"azelaic acid", r"azeloyl"],
    "niacinamide":      [r"niacinamide", r"nicotinamide"],
    "sulfur":           [r"\bsulfur\b", r"\bsulphur\b"],
    "tea_tree":         [r"melaleuca alternifolia", r"tea tree"],
    "zinc":             [r"zinc (pca|gluconate|oxide|sulfate)"],
    "clay_charcoal":    [r"kaolin", r"bentonite", r"montmorillonite",
                         r"charcoal", r"illite", r"\bclay\b"],
    # --- exfoliants ---
    "aha":              [r"glycolic acid", r"lactic acid", r"mandelic acid",
                         r"\balpha hydroxy\b", r"\baha\b", r"malic acid",
                         r"tartaric acid"],
    "pha":              [r"gluconolactone", r"lactobionic acid", r"\bpha\b"],
    # --- pigmentation / PIH (disproportionately affects darker skin) ---
    "vitamin_c":        [r"ascorbic acid", r"ascorbyl", r"ascorbate",
                         r"3-o-ethyl ascorbic"],
    "tranexamic_acid":  [r"tranexamic acid", r"cetyl tranexamate"],
    "alpha_arbutin":    [r"arbutin"],
    "kojic_acid":       [r"kojic acid", r"kojic dipalmitate"],
    "licorice_root":    [r"glycyrrhiza", r"licorice", r"liquorice"],
    # --- barrier / soothing (constraints for dry & sensitive skin) ---
    "ceramide":         [r"ceramide"],
    "hyaluronic_acid":  [r"hyaluron", r"sodium hyaluronate"],
    "glycerin":         [r"\bglycerin\b", r"glycerol"],
    "panthenol":        [r"panthenol", r"pantothenic"],
    "centella":         [r"centella", r"madecassoside", r"asiaticoside",
                         r"\bcica\b"],
    "squalane":         [r"squalane"],
    "colloidal_oatmeal": [r"avena sativa", r"colloidal oat"],
    "allantoin":        [r"allantoin"],
    # --- irritancy flags (used to EXCLUDE for sensitive/dry declarations) ---
    "denatured_alcohol": [r"alcohol denat", r"\bsd alcohol\b",
                          r"\bethanol\b(?!.*cetyl)"],
    "fragrance":        [r"\bfragrance\b", r"\bparfum\b", r"\bperfume\b"],
    "essential_oil":    [r"citrus .*oil", r"lavandula.*oil", r"mentha.*oil",
                         r"eucalyptus.*oil", r"peppermint oil", r"\bmenthol\b",
                         r"\blimonene\b", r"\blinalool\b", r"\bgeraniol\b"],
    "physical_spf":     [r"titanium dioxide", r"zinc oxide"],
}

IRRITANCY_CLASSES = ["denatured_alcohol", "fragrance", "essential_oil"]
COMPILED = {cls: [re.compile(p, re.IGNORECASE) for p in pats]
            for cls, pats in INGREDIENT_CLASSES.items()}


def normalise_ingredients(raw):
    """Sephora stores ingredients as a stringified list, sometimes nested,
    sometimes with marketing preamble. Return one lowercase string."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = str(raw).strip()
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                s = " , ".join(str(x) for x in parsed)
        except (ValueError, SyntaxError):
            s = s.strip("[]")
    s = s.replace("\n", " ").replace("*", " ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def flag_classes(text):
    return {cls: int(any(p.search(text) for p in pats))
            for cls, pats in COMPILED.items()}


# ============================================================ 2. PARSE
prod = pd.read_parquet(IN)
prod["ing_norm"] = prod["ingredients"].map(normalise_ingredients)
flags = pd.DataFrame([flag_classes(t) for t in prod["ing_norm"]],
                     index=prod.index)
prod = pd.concat([prod, flags], axis=1)
prod["has_ingredients"] = (prod["ing_norm"].str.len() > 20).astype(int)
prod["n_irritants"] = prod[IRRITANCY_CLASSES].sum(axis=1)

# skin-type suitability declared in the marketing `highlights` field
def suits(hl, kind):
    s = str(hl).lower()
    return int(kind in s)


prod["hl_dry"] = prod["highlights"].map(lambda h: suits(h, "dry"))
prod["hl_oily"] = prod["highlights"].map(lambda h: suits(h, "oily"))
prod["hl_combo"] = prod["highlights"].map(lambda h: suits(h, "combo"))
prod["hl_sensitive"] = prod["highlights"].map(lambda h: suits(h, "sensitiv"))

os.makedirs(PROC, exist_ok=True)
keep = (["product_id", "product_name", "brand_name", "price_usd", "rating",
         "n_reviews", "mean_rating", "pct_recommended", "pct_positive",
         "loves_count", "primary_category", "secondary_category",
         "tertiary_category", "has_ingredients", "n_irritants",
         "hl_dry", "hl_oily", "hl_combo", "hl_sensitive",
         "reviews_combination", "reviews_dry", "reviews_normal", "reviews_oily"]
        + list(INGREDIENT_CLASSES.keys()))
prod[keep].to_parquet(os.path.join(PROC, "product_ingredient_flags.parquet"),
                      index=False)


# ============================================================ 3. REPORT
w("# Ingredient Parsing and the Severity-to-Ingredient Mapping Table\n")
w("_Proposal sections 7.3 and 7.4. This table is the interpretable seam "
  "between the image module and the recommender: every recommendation is "
  "traceable through it, which is what makes the pipeline auditable._\n")

w("## 1. Parsing coverage\n")
w(f"- Skincare products in catalogue: **{len(prod):,}**")
w(f"- With a usable ingredient list: **{int(prod['has_ingredients'].sum()):,}** "
  f"({100*prod['has_ingredients'].mean():.1f}%)")
w(f"- Controlled vocabulary size: **{len(INGREDIENT_CLASSES)}** ingredient classes\n")

w("**Products matched per ingredient class:**\n")
w("| ingredient class | products | % of catalogue |")
w("|---|---|---|")
for cls in INGREDIENT_CLASSES:
    n = int(prod[cls].sum())
    w(f"| `{cls}` | {n:,} | {100*n/len(prod):.1f}% |")
w("")

# the imbalance Lee et al. (2024) report -- acne-relevant products are scarce
acne_actives = ["salicylic_acid", "benzoyl_peroxide", "retinoid",
                "azelaic_acid", "sulfur"]
n_acne = int((prod[acne_actives].sum(axis=1) > 0).sum())
n_aging = int((prod[["retinoid", "vitamin_c", "hyaluronic_acid"]].sum(axis=1) > 0).sum())
w(f"**Corpus-level observation (motivates RQ5):** only **{n_acne:,}** products "
  f"({100*n_acne/len(prod):.1f}%) carry any acne-directed active, against "
  f"**{n_aging:,}** carrying an anti-ageing/hydration active. This mirrors the "
  "concern-labelling imbalance Lee et al. (2024) report and bounds how much "
  "the recommender can differentiate at severe grades.\n")

w("## 2. The mapping table (severity -> ingredient classes)\n")
w("ACNE04 grades follow the Hayashi criterion. Weights are *preferences*, not "
  "doses: they score candidate products, they do not prescribe.\n")
w("| grade | severity | preferred ingredient classes | de-prioritised | referral |")
w("|---|---|---|---|---|")
w("| 0 | Mild | `salicylic_acid`, `niacinamide`, `aha` (gentle), `pha` | high-strength `benzoyl_peroxide` | no |")
w("| 1 | Moderate | `salicylic_acid`, `benzoyl_peroxide`, `retinoid`, `azelaic_acid`, `niacinamide` | heavy occlusives | no |")
w("| 2 | Severe | `benzoyl_peroxide`, `retinoid`, `azelaic_acid`, `niacinamide` | pure hydrators alone | **yes** |")
w("| 3 | Very severe | `benzoyl_peroxide`, `retinoid`, `azelaic_acid` | cosmetic-only actives | **yes** |")
w("")
w("**Declared-attribute modifiers** (elicited conversationally, section 7.7 -- "
  "never inferred from pixels):\n")
w("| declared attribute | pulls toward | pushes away from |")
w("|---|---|---|")
w("| Dry | `ceramide`, `hyaluronic_acid`, `glycerin`, `squalane`, `panthenol` | `denatured_alcohol`, high-strength exfoliants |")
w("| Sensitive | `centella`, `colloidal_oatmeal`, `allantoin`, `panthenol` | `fragrance`, `essential_oil`, `denatured_alcohol` |")
w("| Oily | `clay_charcoal`, `zinc`, `niacinamide`, `salicylic_acid` | heavy occlusives |")
w("| Combination | `niacinamide`, `pha`, lightweight hydrators | strong solvents |")
w("| PIH / dark marks | `niacinamide`, `azelaic_acid`, `vitamin_c`, `tranexamic_acid`, `alpha_arbutin` | irritants that worsen inflammation |")
w("")
w("The PIH row matters for this project specifically: post-inflammatory "
  "hyperpigmentation disproportionately affects darker skin, and RQ5 found "
  "Deep-tone reviewers raise it at 6.7% versus 2.9% for Light-tone reviewers "
  "while contributing only 2.9% of reviews.\n")

w("## 3. Safety position\n")
w("- The system is a **cosmetic recommendation tool, not a medical device**. "
  "Grades 2 and 3 attach a referral flag; the interface states that "
  "persistent or severe acne warrants qualified care.")
w("- No prescription-strength agent (e.g. tretinoin, oral therapy) is ever "
  "recommended; `retinoid` here means cosmetic retinol-class ingredients "
  "available in the retail catalogue.")
w("- Irritancy classes are used only to *exclude* products against a declared "
  "sensitivity, never to make a claim about a product's safety in general.\n")

w("## 4. Why rule-based\n")
w("The ingredient field is inconsistent free text with marketing preamble, "
  "nested list encodings, and variable ordering. A deterministic parser makes "
  "every match inspectable and every recommendation explainable -- a learned "
  "extractor would obscure exactly the seam the fairness analysis needs to "
  "examine. Conservative patterns are preferred: a false negative costs a "
  "missed candidate, a false positive puts an unsuitable product in front of "
  "a user.\n")

with open(OUT_MD, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
w(f"Saved: {PROC}/product_ingredient_flags.parquet")
w(f"Saved: {OUT_MD}")
