# Review Sentiment Model (proposal §7.3)

_Results from `kaggle/nb7_sentiment_model.ipynb`: DistilBERT-base fine-tuned on
120,000 label-balanced Sephora reviews (60,000 per class), 2 epochs, held-out
test set of 24,000._

## 1. Why this exists

The proposal commits to sentiment classification by a fine-tuned transformer.
All earlier analysis in this project used a **proxy** — treating `rating >= 4`
as positive. The proxy is standard and defensible, but it is not what the
proposal specifies, and it cannot detect sentiment carried in text that
diverges from the star rating. This notebook closes that gap and, separately,
tests whether the text model reads all reviewers equally well.

**Label construction.** Labels derive from star ratings (standard weak
supervision). **3-star reviews are dropped as ambiguous** rather than forced
into a class. The corpus is 88.7% positive, so the training sample was balanced
to 50/50 to prevent the model from learning the prior instead of the language.

## 2. Held-out performance

| metric | value |
|---|---|
| accuracy | **0.9743** |
| macro-F1 | **0.9743** |
| ROC-AUC | **0.9959** |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| negative | 0.97 | 0.98 | 0.97 | 12,000 |
| positive | 0.98 | 0.97 | 0.97 | 12,000 |

Confusion matrix: 11,720 / 280 / 336 / 11,664. Errors are near-symmetric across
classes, so the model is not trading one class off against the other.

## 3. Does the model read all reviewers equally well?

This is the fairness extension to RQ5. If the text model were less accurate on
reviews written by one tone group, that group's expressed preferences would be
measured less reliably — a bias entering the recommender through the *text*
pipeline rather than through the image model.

| tone band | n | accuracy [95% CI] | macro-F1 |
|---|---|---|---|
| Light | 15,887 | 0.9751 [0.9726, 0.9773] | 0.9751 |
| Medium | 3,630 | 0.9774 [0.9725, 0.9821] | 0.9774 |
| Deep | 575 | 0.9704 [0.9565, 0.9827] | 0.9704 |

**Accuracy spread: 0.0070 — below any material threshold, with heavily
overlapping intervals. No disparity is demonstrated.** The text pipeline reads
reviews from all three tone bands with effectively equal accuracy.

Two qualifications matter. First, the Deep band contributes only **575 test
reviews**, so its interval is roughly five times wider than the Light band's
(±1.3pp against ±0.24pp); a disparity smaller than about two percentage points
would not be detectable there. Second — and this is the more important point —
**the sample sizes themselves reproduce the corpus imbalance documented in
RQ5**. Even when a downstream model is even-handed, the underlying data is not,
and it is that imbalance, rather than model behaviour, that limits what can be
established about the smallest group.

## 4. What the model tells us about the rating proxy

**A correction to the notebook's own framing.** The notebook reports
"text-model / star-rating agreement: 0.9743" and describes the 2.6% complement
as "the share of the corpus the rating proxy was getting wrong". **That
interpretation is not valid and should not be carried into the dissertation.**

Because the labels *are* the rating threshold, agreement with the rating is
identical to accuracy by construction — the two numbers are the same quantity
reported twice. The 2.6% is the model's error rate against rating-derived
labels; it is not independent evidence that the rating mislabelled those
reviews. Distinguishing "the text disagrees with the stars" from "the model is
wrong" would require human-labelled sentiment on a held-out subset, which this
project does not have.

The defensible finding is the **product-level correlation**, which is
independent of the labelling scheme:

> Across 2,255 products, text-derived positive rate and rating-derived positive
> rate correlate at **r = 0.904**.

That figure supports a useful retrospective conclusion: **the rating proxy used
throughout the earlier analysis was a sound stand-in.** Roughly 82% of
product-level variance in text sentiment is shared with the rating proxy, so
substituting the transformer would not have materially changed the RQ2
recommender rankings or the RQ5 aggregates. The remaining ~18% is where
text-based sentiment carries information the stars do not.

This is a validation result rather than a corrective one, and it is worth
reporting as such: the earlier analysis is strengthened by the check, not
overturned by it.

## 5. Artefacts

- `sentiment_model/` — fine-tuned DistilBERT weights and tokeniser
- `sentiment_test_predictions.csv` — per-review test predictions with tone band
- `sentiment_by_tone.csv` — the disaggregated table above
- `product_sentiment_scores.csv` — per-product text sentiment for 2,255
  products (drop-in replacement for `pct_positive` in `scripts/recommender.py`)
- `sentiment_results.png` — accuracy by tone; text sentiment vs rating proxy
- `sentiment_result.json` — machine-readable summary

## 6. Limitations

- **Labels are rating-derived**, not human-annotated. The model learns to
  recover the star rating from text, which is the standard setup but means its
  errors cannot be cleanly attributed between model and label.
- **Only 200,000 reviews were scored** for the per-product export (of
  1,094,411), for runtime reasons. Products with few scored reviews have noisy
  sentiment estimates; `n_scored` is retained in the export so they can be
  filtered.
- The **Deep band test sample (575)** is small enough that only large
  disparities would be detectable.
