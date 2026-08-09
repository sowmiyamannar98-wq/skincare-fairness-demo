# Skincare Assistant — research demonstrator

**This is a cosmetic product recommendation tool, not a medical device. It does
not diagnose.** For persistent, painful or worsening skin problems, please see a
qualified clinician.

The artefact accompanying an MSc Data Science dissertation, *Does Bias
Propagate? Fairness in Automated Acne Severity Assessment and Downstream
Skincare Recommendation.* It exists to make the measured pipeline inspectable,
not to give skincare advice.

## What it does

1. A ResNet-50 classifier assesses acne severity from an uploaded photo on a
   four-point ordinal scale. It **abstains** below a confidence threshold of
   0.85 rather than emitting a low-confidence guess.
2. A rules-based conversation collects what an image cannot supply: skin type,
   sensitivity, dark-mark concern and budget.
3. A documented mapping table converts the grade and those attributes into
   preferred and de-prioritised ingredient classes.
4. A deterministic ranker scores the catalogue on ingredient match, review
   sentiment, star rating, skin-type fit, and an irritancy penalty applied only
   when sensitivity is declared.

## The architectural constraint

**There is no language model anywhere in this application.** That is deliberate
and is the point of the artefact.

| Component | Responsibility | Determines recommendations? |
|---|---|---|
| Dialogue manager | Question phrasing, parsing replies into slots | No |
| CNN classifier | Acne severity from the image | Yes |
| Mapping table | Severity and attributes to ingredient classes | Yes |
| Ranking function | Product scoring and ordering | Yes |

A generative component sitting between the audited classifier and the user
would mean divergence between recommendation lists could no longer be
attributed to the classifier, and the dissertation's propagation measurement
would no longer characterise the system it claims to measure. Every
recommendation traces deterministically from classifier, through the mapping
table, to the ranking function.

Quick-reply buttons are a keyboard shortcut only: each submits ordinary text
through the same parser a typed reply uses.

## Safety behaviour

Medical requests — diagnosis, malignancy assessment, prescription guidance,
emergency symptoms and self-treatment procedures — are refused and redirected to
qualified care. The behaviour is specified in advance in `app/safety.py`, not
delegated to a model's discretion, and was tested against a purpose-built set of
21 adversarial requests alongside 12 benign controls. The final evaluation
returns a refusal rate of 1.00 and a false-refusal rate of 0.00. Severity grades
2 and 3 additionally attach a referral flag.

## Known limitations

Documented in full in the dissertation, and they matter for interpreting
anything this demonstrator outputs.

- The severity model was trained on ACNE04, whose population is predominantly
  Chinese and which contains **no auditable population of dark-skinned
  subjects**. Behaviour on darker skin is not validated.
- A counterfactual probe found the model is **causally sensitive to skin tone**:
  6.8% of predictions change grade under a 20-degree darkening shift with lesion
  content held fixed, and instability is greatest where training tone coverage
  is thinnest.
- The catalogue is prestige-only, with no mass-market products, so the price
  range is not representative.
- Out-of-fold accuracy is 0.766, macro-F1 0.746; the severe grade is weakest at
  F1 0.62.

## Privacy

Uploaded images are processed in memory to produce a severity grade and are
neither written to disk nor retained. If this is deployed publicly, do not
upload images of anyone who has not consented.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Deploying to Streamlit Community Cloud

Push this repository to GitHub, then at [share.streamlit.io](https://share.streamlit.io)
choose **New app**, select the repository, and set:

- **Main file path**: `app/app.py`
- **Python version**: 3.11

`requirements.txt` pins CPU-only torch wheels deliberately. The default PyPI
torch bundles CUDA and is several gigabytes, which exceeds the free tier's
resource limit. torch is pinned to the version that wrote the checkpoint; an
older torch may fail to read it, so re-test loading before relaxing that pin.

The model checkpoint is 90 MiB, under GitHub's 100 MiB per-file limit, so Git
LFS is not required.

## Attribution

The product catalogue derives from the public *Sephora Products and Skincare
Reviews* dataset on Kaggle. The severity model was trained on ACNE04 (Wu et al.,
ICCV 2019). Neither source dataset is redistributed here; this repository
contains only a derived table of product metadata and parsed ingredient-class
flags required for the ranker to run.
