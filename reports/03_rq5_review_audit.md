# RQ5 — Representational Audit of the Sephora Review Corpus

_Independent of the image pipeline. Reviewer skin tone is self-reported; tones are collapsed into three ordered bands (Light / Medium / Deep). Reviews with missing or non-committal tone (`notSureST`, blank) are excluded from disaggregated statistics and counted separately._

- Total reviews scanned: **1,094,411**
- Reviews with a usable tone band: **923,802** (84.4%)
- Missing skin_tone: **170,539** (15.6%)
- `notSureST` (declined): **70** (0.0%)

## 1. Representation (review volume by tone band)

| band | reviews | share | 95% CI on share |
|---|---|---|---|
| Light | 728,833 | 78.9% | [78.8%, 79.0%] |
| Medium | 168,350 | 18.2% | [18.1%, 18.3%] |
| Deep | 26,619 | 2.9% | [2.8%, 2.9%] |

**Representation ratio (Light : Deep) = 27.4 : 1.** The corpus is dominated by lighter-tone reviewers; every downstream per-tone estimate for the Deep band rests on a far thinner sample, which is why intervals are reported throughout.

## 2. Rating, sentiment, and price by tone (bootstrapped 95% CIs)

| band | mean rating | P(recommended) | P(rating>=4) | median price (USD) |
|---|---|---|---|---|
| Light | 4.294 [4.292, 4.297] | 83.8% [83.7, 83.9] | 81.9% [81.8, 81.9] | $40 [$40, $40] |
| Medium | 4.303 [4.297, 4.308] | 84.5% [84.4, 84.7] | 82.4% [82.2, 82.5] | $39 [$39, $39] |
| Deep | 4.267 [4.252, 4.280] | 84.9% [84.5, 85.3] | 81.8% [81.3, 82.2] | $38 [$38, $39] |

## 3. Significance tests and effect sizes

- **Rating ~ tone (Kruskal-Wallis):** H=63.5, p=1.62e-14, epsilon-squared=0.0001 (negligible effect). A significant p on ~1M rows is expected; the effect size is the honest measure of practical magnitude.
- **Price ~ tone (Kruskal-Wallis):** H=409.2, p=1.40e-89, epsilon-squared=0.0004.
- **PIH-concern mention ~ tone (chi-square):** chi2=2347.5, p=0.00e+00, Cramer's V=0.0504.

## 4. Concern coverage — post-inflammatory hyperpigmentation family

Share of each band's reviews that mention hyperpigmentation / dark-spot / acne-scarring concerns. These concerns disproportionately affect darker skin, so their prevalence by tone speaks to whether the corpus surfaces the needs of darker-skinned users.

| band | reviews mentioning PIH concerns | rate | 95% CI |
|---|---|---|---|
| Light | 20,748 | 2.85% | [2.81%, 2.89%] |
| Medium | 7,726 | 4.59% | [4.48%, 4.70%] |
| Deep | 1,794 | 6.74% | [6.45%, 7.04%] |

## 5. Figures

- `reports/figures/rq5_representation.png`
- `reports/figures/rq5_rating_ci.png`
- `reports/figures/rq5_price_ci.png`
- `reports/figures/rq5_pih_coverage.png`

## 6. Reading these results

- **Representation is the dominant finding.** The Light:Deep volume ratio is 27:1; darker-skinned reviewers are severely under-represented, so the platform's aggregate review signal is overwhelmingly a lighter-skin signal. This is a representational bias in the corpus itself, before any model is trained on it.
- **Effect sizes, not p-values, carry the argument.** With ~1M reviews, almost any difference is 'significant'; epsilon-squared and Cramer's V are reported so that trivial differences are not oversold.
- **Concern coverage** indicates whether darker-skin concerns (PIH) are even voiced in the corpus at rates matching their real-world prevalence; a low absolute Deep-band volume means even a higher per-review rate translates into few reviews the recommender can learn from.
- **Independence.** None of this depends on the image model; it is a self-contained contribution answering RQ5.

