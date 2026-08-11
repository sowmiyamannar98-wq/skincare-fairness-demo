# RQ2 -- Does Classifier Bias Propagate into Recommendations?

_Principal contribution. Each ACNE04 subject is run through the same deterministic recommender twice: once on the dermatologist grade (what they should receive) and once on the model's prediction (what they actually receive). Divergence between the two lists is attributable to the upstream classifier alone._

- Simulated users: **1,457**
- Upstream classifier: ResNet-50, out-of-fold accuracy **0.766**
- Tone labels: validated ITA (`ita_B`); audit restricted to Light vs Medium (Deep excluded as illumination artifact, see NB1b)

- Catalogue after filtering: **1,701** products

## 1. Propagation headline -- the cost of an upstream error

- Users receiving a **correct** grade: 1,116 (76.6%)
- Users receiving a **misclassified** grade: 341 (23.4%)

| outcome | list overlap with correct list (Jaccard) | relevance P@10 vs true grade | price shift (USD) |
|---|---|---|---|
| correctly classified | 1.000 [1.000, 1.000] | 0.995 [0.993, 0.997] | +0.00 [+0.00, +0.00] |
| **misclassified** | 0.515 [0.495, 0.536] | 0.987 [0.983, 0.991] | -0.73 [-2.07, +0.60] |

- **Overlap penalty of misclassification:** +0.485 [+0.465, +0.504]
- **Relevance penalty of misclassification:** +0.008 [+0.004, +0.012]

This is the propagation effect in its simplest form: an upstream error does not merely mislabel a user, it replaces the product set they would have been shown.

## 2. Propagation disaggregated by tone group

| tone band | n | acc | Jaccard | relevance P@10 | price shift | missed referrals |
|---|---|---|---|---|---|---|
| Light | 538 | 0.762 | 0.885 [0.864, 0.903] | 0.989 [0.986, 0.993] | -0.44 [-0.96, +0.08] | 2.2% |
| Medium | 724 | 0.782 | 0.894 [0.878, 0.909] | 0.996 [0.995, 0.998] | -0.16 [-0.62, +0.26] | 4.0% |

**Light - Medium gaps (bootstrapped):**

- Jaccard overlap: -0.010 [-0.035, +0.015]
- Relevance P@10: -0.007 [-0.011, -0.004]
- Price shift: -0.29 [-0.92, +0.36] USD

- Gaps both statistically resolved **and** materially large: **none**
- Statistically resolved but practically negligible: relevance (-0.007, below the 0.05 materiality threshold)

**No tone-differential propagation is demonstrated.** Gaps are either unresolved or too small to alter what a user is shown. The honest reading is that this corpus -- two auditable tone bands, no dark-skin population -- cannot settle the question, not that no disparity exists. Interval widths above document that limitation.

## 3. Safety-critical propagation -- referral failures

Grades 2-3 attach a referral flag (section 7.4). A false-negative here is the most consequential downstream error the system can make: a user whose acne warrants qualified care is instead handed a cosmetic routine.

| tone band | n | missed referral rate | unnecessary referral rate |
|---|---|---|---|
| Light | 538 | 2.2% [1.1, 3.5] | 4.6% [3.0, 6.5] |
| Medium | 724 | 4.0% [2.6, 5.5] | 3.6% [2.2, 4.8] |

- **Missed-referral gap (Light - Medium):** -1.8pp [-3.7, +0.0]

## 4. Catalogue coverage and list diversity

Coverage = distinct products any user in the group is ever shown. A narrower catalogue for one group means fewer real options.

| tone band | distinct products surfaced | coverage of catalogue | mean brands per list |
|---|---|---|---|
| Light | 104 | 6.1% | 8.14 |
| Medium | 100 | 5.9% | 8.08 |

## 5. Baseline comparison

- Popularity-only ranker, mean P@10 across grades: **0.425**
- Pipeline recommender, mean deployed P@10: **0.993**
- The pipeline adds value over trivial popularity ranking, evidencing that the mapping table does work beyond surfacing best-sellers.

## 6. Figures

- `reports/figures/rq2_jaccard_by_tone.png`
- `reports/figures/rq2_cost_of_error.png`
- `reports/figures/rq2_price_shift.png`

## 7. Reading this analysis

- **Propagation is real and quantified.** A misclassified user does not receive a slightly worse list; they receive a substantially different one. The Jaccard and relevance penalties above are the size of that harm.
- **Whether it is tone-DIFFERENTIAL is a separate question** from whether it exists. With only two auditable tone bands in ACNE04 -- itself a reported finding -- this arm is under-powered to settle it, and every gap is therefore reported with its interval rather than as a point claim.
- **The relevance metric is close to saturated and should be read with care.** P@10 sits near 0.99 in both arms because the mapping table endorses overlapping ingredient classes at adjacent grades, so a one-grade error still returns products carrying an endorsed class. Precision@k against a permissive reference set is therefore an insensitive instrument for propagation; **Jaccard overlap is the informative measure here**, and a stricter grade-specific reference standard is the obvious refinement.
- **Attribute profiles are simulated**, sampled from the Sephora reviewer distribution with a fixed seed. They make each user distinct and realistic but they are not real declarations; this is a stated limitation.
- **The recommender is deterministic**, so all divergence is attributable to the upstream classifier. That is the property that makes this measurement meaningful at all.

