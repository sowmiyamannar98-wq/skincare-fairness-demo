# Ingredient Parsing and the Severity-to-Ingredient Mapping Table

_Proposal sections 7.3 and 7.4. This table is the interpretable seam between the image module and the recommender: every recommendation is traceable through it, which is what makes the pipeline auditable._

## 1. Parsing coverage

- Skincare products in catalogue: **2,420**
- With a usable ingredient list: **2,281** (94.3%)
- Controlled vocabulary size: **28** ingredient classes

**Products matched per ingredient class:**

| ingredient class | products | % of catalogue |
|---|---|---|
| `salicylic_acid` | 325 | 13.4% |
| `benzoyl_peroxide` | 7 | 0.3% |
| `retinoid` | 164 | 6.8% |
| `azelaic_acid` | 31 | 1.3% |
| `niacinamide` | 351 | 14.5% |
| `sulfur` | 11 | 0.5% |
| `tea_tree` | 36 | 1.5% |
| `zinc` | 210 | 8.7% |
| `clay_charcoal` | 150 | 6.2% |
| `aha` | 423 | 17.5% |
| `pha` | 129 | 5.3% |
| `vitamin_c` | 627 | 25.9% |
| `tranexamic_acid` | 25 | 1.0% |
| `alpha_arbutin` | 11 | 0.5% |
| `kojic_acid` | 8 | 0.3% |
| `licorice_root` | 293 | 12.1% |
| `ceramide` | 230 | 9.5% |
| `hyaluronic_acid` | 1,043 | 43.1% |
| `glycerin` | 1,797 | 74.3% |
| `panthenol` | 351 | 14.5% |
| `centella` | 125 | 5.2% |
| `squalane` | 550 | 22.7% |
| `colloidal_oatmeal` | 164 | 6.8% |
| `allantoin` | 229 | 9.5% |
| `denatured_alcohol` | 207 | 8.6% |
| `fragrance` | 762 | 31.5% |
| `essential_oil` | 900 | 37.2% |
| `physical_spf` | 414 | 17.1% |

**Corpus-level observation (motivates RQ5):** only **473** products (19.5%) carry any acne-directed active, against **1,339** carrying an anti-ageing/hydration active. This mirrors the concern-labelling imbalance Lee et al. (2024) report and bounds how much the recommender can differentiate at severe grades.

## 2. The mapping table (severity -> ingredient classes)

ACNE04 grades follow the Hayashi criterion. Weights are *preferences*, not doses: they score candidate products, they do not prescribe.

| grade | severity | preferred ingredient classes | de-prioritised | referral |
|---|---|---|---|---|
| 0 | Mild | `salicylic_acid`, `niacinamide`, `aha` (gentle), `pha` | high-strength `benzoyl_peroxide` | no |
| 1 | Moderate | `salicylic_acid`, `benzoyl_peroxide`, `retinoid`, `azelaic_acid`, `niacinamide` | heavy occlusives | no |
| 2 | Severe | `benzoyl_peroxide`, `retinoid`, `azelaic_acid`, `niacinamide` | pure hydrators alone | **yes** |
| 3 | Very severe | `benzoyl_peroxide`, `retinoid`, `azelaic_acid` | cosmetic-only actives | **yes** |

**Declared-attribute modifiers** (elicited conversationally, section 7.7 -- never inferred from pixels):

| declared attribute | pulls toward | pushes away from |
|---|---|---|
| Dry | `ceramide`, `hyaluronic_acid`, `glycerin`, `squalane`, `panthenol` | `denatured_alcohol`, high-strength exfoliants |
| Sensitive | `centella`, `colloidal_oatmeal`, `allantoin`, `panthenol` | `fragrance`, `essential_oil`, `denatured_alcohol` |
| Oily | `clay_charcoal`, `zinc`, `niacinamide`, `salicylic_acid` | heavy occlusives |
| Combination | `niacinamide`, `pha`, lightweight hydrators | strong solvents |
| PIH / dark marks | `niacinamide`, `azelaic_acid`, `vitamin_c`, `tranexamic_acid`, `alpha_arbutin` | irritants that worsen inflammation |

The PIH row matters for this project specifically: post-inflammatory hyperpigmentation disproportionately affects darker skin, and RQ5 found Deep-tone reviewers raise it at 6.7% versus 2.9% for Light-tone reviewers while contributing only 2.9% of reviews.

## 3. Safety position

- The system is a **cosmetic recommendation tool, not a medical device**. Grades 2 and 3 attach a referral flag; the interface states that persistent or severe acne warrants qualified care.
- No prescription-strength agent (e.g. tretinoin, oral therapy) is ever recommended; `retinoid` here means cosmetic retinol-class ingredients available in the retail catalogue.
- Irritancy classes are used only to *exclude* products against a declared sensitivity, never to make a claim about a product's safety in general.

## 4. Why rule-based

The ingredient field is inconsistent free text with marketing preamble, nested list encodings, and variable ordering. A deterministic parser makes every match inspectable and every recommendation explainable -- a learned extractor would obscure exactly the seam the fairness analysis needs to examine. Conservative patterns are preferred: a false negative costs a missed candidate, a false positive puts an unsuitable product in front of a user.

