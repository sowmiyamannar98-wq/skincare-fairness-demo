# Results — draft chapter

> **How to use this.** Every number here is real and traceable to a script or
> notebook in this repository. Prose is drafted, not skeletal: edit for voice
> and length rather than starting from blank. Figure callouts name files in
> `reports/figures/` or Kaggle outputs. Sections map one-to-one onto the six
> research questions, per the proposal's commitment that the results chapter be
> structured around them.
>
> **Figures still to download from Kaggle:** `rq3_tradeoff_curve.png`,
> `rq4_tone_sensitivity.png`, `rq4_shift_example.png`, `arch_comparison.png`,
> `ita_montage_method_{A,B,C}.png`, `ita_method_distributions.png`.

---

## 5.1 Corpus characterisation and the skin-tone gate

### 5.1.1 The image corpus

ACNE04 comprises 1,457 dermatologist-graded facial photographs under the
Hayashi criterion. The severity distribution is imbalanced: grade 0 (mild)
497 images, grade 1 (moderate) 637, grade 2 (severe) 186, grade 3 (very severe)
137. The two severe grades together account for 22.2% of the corpus, and this
imbalance propagates into every result that follows — most visibly into the
per-class performance reported in §5.2.

### 5.1.2 Deriving skin tone, and why the first attempt failed

Neither facial corpus supplies Fitzpatrick labels, so tone groups were derived
computationally using the Individual Typology Angle (ITA) in CIELAB space,
following established practice in the dermatological fairness literature.

The initial implementation — a permissive skin mask with percentile trimming on
luminance — returned a distribution in which **59.5% of ACNE04 fell into the
darkest band**. ACNE04's population is predominantly Chinese, so this result is
not credible on its face, and it was treated as a measurement failure rather
than a finding.

Validation followed the procedure committed to in the proposal: manual
inspection of a stratified sample. Montages of randomly sampled faces grouped
by estimated tone (Figure X: `ita_montage_method_A.png`) make the failure
immediately visible, and two distinct error modes are apparent.

**The darkest band contains ordinary subjects.** The faces assigned to the
"Deep" band under Method A (ITA 1–18°) are unremarkable light-to-medium skinned
faces, several of them evenly lit. They are not dark-skinned subjects by any
reasonable visual judgement. The estimator is not mis-sorting a genuine
minority; it is systematically under-reading ITA across the corpus, because
averaging over all skin-masked pixels incorporates shadowed peripheral regions,
hair boundaries, and dim ambient lighting, all of which depress median
luminance.

**Inflamed close-ups are pushed the other way.** The "Light" band under Method
A (ITA 44–62°) is dominated by tightly cropped, heavily inflamed cheek shots.
Erythema raises the a\* and b\* channels, and because ITA is computed as
arctan((L\*−50)/b\*), inflammation inflates the estimate. Severity therefore
contaminates the tone measurement in the direction that would most damage a
fairness audit: the most severe cases are systematically assigned to one tone
band.

Together these confirm the distribution is a measurement artefact rather than a
property of the corpus, and they identify the two confounds a corrected
estimator must remove.

Two refinements were tested against the same visual standard:

| method | pixel selection | median ITA | Light / Medium / Deep |
|---|---|---|---|
| A (initial) | skin mask, luminance trimmed | 14.8° | 4% / 36% / 59% |
| **B (adopted)** | drop reddest 25%, keep brighter half | **37.3°** | 37% / 50% / 13% |
| C | central crop, then B | 45.8° | 67% / 26% / 7% |

Method B removes the two identified confounds directly: discarding the reddest
quartile of skin pixels suppresses the contribution of inflamed lesions, and
retaining the brighter half of the remaining distribution suppresses shadow. It
returns a median ITA of 37.3°, corresponding to Fitzpatrick type III — which is
consistent with the corpus demographic.

The distributions themselves (Figure X: `ita_method_distributions.png`) supply
a further check. All three methods retain a left tail extending below −50°.
Values that extreme are not plausible skin measurements for any subject in this
corpus, and their persistence across estimators indicates a floor of
irreducible measurement noise arising from illumination rather than from
pigmentation. Method B compresses the bulk of the distribution into a plausible
range around Fitzpatrick III while leaving that tail largely intact, which is
consistent with the tail being an artefact of image capture that no
pixel-selection rule can remove. Method B was adopted for all subsequent
analysis; Method C is retained as a sensitivity variant. Agreement between
methods B and C on band assignment is 60.1%, which is itself a caution against
treating any single ITA implementation as authoritative.

### 5.1.3 The go/no-go gate: a compressed tone range

The proposal committed to an explicit decision gate before modelling, on the
grounds that ACNE04's tone range is load-bearing for RQ1 and RQ2.

Under the corrected estimator the finding is unambiguous and is reported here
as a substantive result rather than as an obstacle: **ACNE04 contains no
population of genuinely dark-skinned subjects.** Under Method B the darkest
band holds 13% of images, but montage inspection confirms that this residual is
composed of the same lighting and inflammation artefacts identified in §5.1.2
rather than of Fitzpatrick V–VI skin. The finding is robust to estimator
choice: it holds under all three methods tested.

The gate outcome is therefore **no-go for the primary path**, and the
pre-committed contingency applies. Three consequences follow, and they shape
the remainder of this chapter:

1. All tone-disaggregated analysis is restricted to the **Light and Medium**
   bands. The darkest band is excluded on measurement-validity grounds, with
   the montages as evidence.
2. Arm B is treated as an **under-powered applied replication**, not as the
   primary audit.
3. The compressed tone range is reported as a **quantified finding about the
   state of publicly available facial acne corpora** — a corpus widely used for
   automated acne grading cannot support a fairness audit across the tone range
   such systems will encounter in deployment.

### 5.1.4 The review corpus

The Sephora corpus contains 8,494 products, of which 2,420 are skincare, and
1,094,411 reviews. Reviewer-reported skin tone is available for 923,802 reviews
(84.4%); 15.6% is missing. Reviewer tone is collapsed into the same three
ordered bands used for the image analysis.

Ingredient parsing over the skincare catalogue yields a corpus-level
observation that bounds what the recommender can achieve: only **473 products
(19.5%)** carry any acne-directed active ingredient, against **1,339 (55.3%)**
carrying an anti-ageing or hydration active. This mirrors the concern-labelling
imbalance reported by Lee et al. (2024) and constrains differentiation at the
severe grades, where the endorsed ingredient classes are narrowest.

---

## 5.2 The severity classifier

A ResNet-50 pretrained on ImageNet was fine-tuned for four-class severity
classification. Given the corpus size, evaluation uses repeated stratified
five-fold cross-validation so that every image contributes an out-of-fold
prediction, rather than a single held-out split of 292 images.

**Out-of-fold performance: accuracy 0.766, macro-F1 0.746**, stable across
folds (macro-F1 0.723–0.774).

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| 0 (mild) | 0.81 | 0.83 | 0.82 | 497 |
| 1 (moderate) | 0.79 | 0.75 | 0.77 | 637 |
| 2 (severe) | 0.58 | 0.66 | **0.62** | 186 |
| 3 (very severe) | 0.79 | 0.77 | 0.78 | 137 |

Performance tracks class frequency, with grade 2 weakest. Class weighting
during training appears to have protected grade 3 — the rarest class — at the
expense of grade 2, its neighbour in an ordinal scale where adjacent-class
confusion is the dominant error mode.

### 5.2.1 Architecture comparison

The proposal committed to comparing this baseline against EfficientNet-B0/B3
and ViT-Base under an identical protocol. Four architectures × three seeds =
twelve runs (Figure X: `arch_comparison.png`).

| architecture | params (M) | accuracy | macro-F1 | severe F1 | ECE |
|---|---|---|---|---|---|
| **ResNet-50** | 23.5 | **0.753 ± 0.037** | **0.731 ± 0.026** | **0.582** | 0.108 |
| EfficientNet-B0 | 4.0 | 0.715 ± 0.040 | 0.690 ± 0.039 | 0.549 | 0.190 |
| EfficientNet-B3 | 10.7 | 0.709 ± 0.017 | 0.679 ± 0.030 | 0.509 | 0.183 |
| ViT-Base | 85.8 | 0.636 ± 0.047 | 0.639 ± 0.038 | 0.492 | **0.085** |

The spread across architectures (0.092) is approximately three times the pooled
seed standard deviation (0.033), so every non-baseline architecture is worse
than ResNet-50 by more than one standard deviation. **The comparison resolves**,
and it resolves in favour of the baseline.

Two observations merit attention. First, **capacity does not help**: the
largest model is the worst by a wide margin. Vision Transformers lack the
convolutional inductive biases of locality and translation equivariance and
compensate for their absence with data; at 1,093 training images that
compensation is unavailable. On a corpus of this size the binding constraint is
data, not architectural capacity.

Second, **calibration is ordered inversely to accuracy**. ViT-Base is the
best-calibrated model (ECE 0.085) despite being the least accurate, while both
EfficientNets are the worst calibrated (0.190, 0.183) despite outperforming it.
A model may be more frequently wrong yet more honest about its uncertainty.
This matters directly for the deployed system, whose abstention behaviour
(§5.7) depends on confidence being meaningful: selection on accuracy alone
would have degraded abstention without appearing in any accuracy metric.

*Limitation.* All architectures were trained at a single learning rate (3e-4),
selected for the ResNet-50 baseline and held constant so that differences would
be attributable to architecture rather than to tuning. ViT-Base is
conventionally fine-tuned at a lower rate (1e-4 to 5e-5), so part of its
deficit is plausibly attributable to hyperparameter mismatch. The defensible
claim is narrower than architectural inferiority: *under a protocol tuned to
the convolutional baseline, ViT-Base underperforms substantially on this
corpus.*

---

## 5.3 RQ1 — Does classification perform equitably across tone groups?

Out-of-fold predictions were disaggregated by validated (Method B) tone band,
with bootstrapped confidence intervals throughout (2,000 resamples).

| group | n | accuracy [95% CI] | macro-F1 [95% CI] | ECE |
|---|---|---|---|---|
| Light | 538 | 0.762 [0.723, 0.797] | 0.739 [0.696, 0.780] | **0.120** |
| Medium | 724 | 0.782 [0.751, 0.812] | 0.735 [0.684, 0.779] | 0.092 |

**Accuracy gap (Light − Medium): −0.020, 95% CI [−0.070, +0.026].**

The interval straddles zero. **No accuracy disparity is demonstrated between
the two auditable tone bands.** Consistent with the pre-committed protocol,
this is reported as a null result rather than reframed, and the interval width
is itself informative: at these group sizes the study can exclude disparities
larger than roughly seven percentage points but not smaller ones.

The substantive finding lies in calibration rather than accuracy. **Expected
Calibration Error is approximately 30% higher for the Light group (0.120
against 0.092)** — the model is more confidently wrong on the smaller of the
two groups. Accuracy parity therefore conceals a difference in the reliability
of the confidence signal, which is precisely the quantity a deployed system
uses to decide whether to advise at all. This observation motivates the
derivation of the abstention threshold in §5.7 and is a direct argument for
reporting calibration alongside accuracy in fairness audits.

Per-class recall exhibits gaps in both directions (Light is weaker at grade 1,
0.660 against 0.808; stronger at grade 3, 0.841 against 0.655). These per-class
contrasts are not accompanied by confidence intervals and rest on small
per-cell counts; they are reported as observations warranting further
investigation, not as findings.

*Figures:* `armB_fair_accuracy.png`, `armB_fair_ece.png`.

---

## 5.4 RQ2 — Does classifier bias propagate into recommendations?

### 5.4.1 Design

Each of the 1,457 subjects was passed through the same recommender twice: once
on the dermatologist's grade — the recommendation they should have received —
and once on the model's out-of-fold prediction, the recommendation the deployed
system would actually produce. Because the mapping table and ranking function
are wholly deterministic, any divergence between the two lists is attributable
to the upstream classifier alone. This property is what makes the measurement
interpretable, and it is the reason a learned ranker was rejected at the design
stage.

Attributes the image cannot supply — skin type, sensitivity, dark-mark concern,
budget — were sampled from the empirical distribution of self-reported Sephora
reviewer attributes under a fixed seed, giving each simulated user a distinct
and realistic profile.

### 5.4.2 Propagation is real and substantial

| outcome | list overlap (Jaccard) | relevance P@10 | price shift (USD) |
|---|---|---|---|
| correctly classified (n=1,116) | 1.000 [1.000, 1.000] | 0.995 [0.993, 0.997] | +0.00 |
| **misclassified (n=341)** | **0.515 [0.495, 0.536]** | 0.987 [0.983, 0.991] | −0.73 [−2.07, +0.60] |

**Overlap penalty of misclassification: +0.485 [0.465, 0.504].**

A misclassified user does not receive a marginally worse recommendation list;
they receive one that shares **barely half its contents** with the list they
should have received. On a corpus where 23.4% of users are misclassified, this
is the harm the system realises at the point of recommendation rather than at
the point of classification — the mechanism this dissertation set out to
measure.

*Figure:* `rq2_cost_of_error.png`.

### 5.4.3 But propagation is not tone-differential

| tone band | n | accuracy | Jaccard | relevance P@10 | missed referrals |
|---|---|---|---|---|---|
| Light | 538 | 0.762 | 0.885 [0.864, 0.903] | 0.989 [0.986, 0.993] | 2.2% |
| Medium | 724 | 0.782 | 0.894 [0.878, 0.909] | 0.996 [0.995, 0.998] | 4.0% |

Light − Medium gaps: Jaccard −0.010 [−0.035, +0.015]; relevance −0.007
[−0.011, −0.004]; price −0.29 USD [−0.92, +0.36]; missed referrals −1.8pp
[−3.7, 0.0].

Only the relevance gap excludes zero, and at 0.007 it falls far below any
threshold at which the products a user sees would change. Reporting statistical
resolution and practical magnitude separately — as the analysis does
explicitly — **no tone-differential propagation is demonstrated.** With two
auditable bands, no dark-skin population, and an upstream classifier that
itself showed no accuracy disparity, this arm cannot settle the question. The
interval widths document that limitation rather than obscuring it.

Two secondary observations:

- **Catalogue concentration.** Across all simulated users, only ~100 distinct
  products (6% of the eligible catalogue) are ever surfaced. Whatever else the
  system does, it funnels every user toward a narrow slice of the available
  products.
- **The pipeline beats a trivial baseline.** Mean P@10 of 0.993 against 0.425
  for a popularity-only ranker confirms the mapping table contributes beyond
  surfacing best-sellers.

*Limitation.* Precision@k is close to saturated in both arms (~0.99), because
the mapping table endorses overlapping ingredient classes at adjacent severity
grades: a one-grade error still returns products carrying an endorsed class.
Precision against a permissive reference set is therefore an insensitive
instrument for propagation, and **Jaccard overlap is the informative measure
here**. A stricter grade-specific reference standard is the obvious refinement.

*Figures:* `rq2_jaccard_by_tone.png`, `rq2_price_shift.png`.

---

## 5.5 RQ3 — Which mitigation strategies reduce disparity, and at what cost?

Six strategies were compared across three seeds (eighteen runs). Given the RQ1
null, the experiment was scoped in advance to the two deficits the audit did
find: the calibration gap and severe-class weakness.

| strategy | macro-F1 | severe F1 | \|acc gap\| | \|ECE gap\| | worst-group acc |
|---|---|---|---|---|---|
| Group reweighting | 0.743 ± 0.035 | 0.595 | 0.034 | 0.051 | 0.760 |
| ERM (baseline) | 0.735 ± 0.031 | 0.591 | 0.036 | 0.027 | 0.756 |
| Class reweighting | 0.731 ± 0.026 | 0.582 | 0.052 | 0.038 | 0.730 |
| Oversampling | 0.731 ± 0.035 | 0.570 | 0.082 | 0.034 | 0.723 |
| GroupDRO | 0.726 ± 0.011 | 0.531 | 0.033 | **0.014** | 0.752 |
| Focal loss | 0.708 ± 0.036 | 0.578 | 0.070 | 0.071 | 0.692 |

**No strategy demonstrably improves on the unmitigated baseline.** The best
apparent gain is 0.008 against a seed standard deviation of ±0.035 — roughly a
quarter of the noise.

The scale of that noise is worth stating directly, because it is the most
transferable result in this section. Under ERM alone, macro-F1 across three
seeds was **0.702 / 0.740 / 0.765** — a spread of 0.063, **eight times** the
largest apparent mitigation gain. Had this experiment been run at a single
seed, almost any ranking of the six strategies could have been produced.

Three qualified observations survive:

- **GroupDRO** produces the smallest calibration gap (0.014) and by far the
  lowest seed variance (±0.011, a third of the others). Its interval includes
  the baseline, so no improvement is demonstrated — but stability across seeds
  is a practical property worth further seeds.
- **Focal loss is the clearest loser** on every axis: lowest macro-F1, worst
  worst-group accuracy, largest and most volatile calibration gap.
  Down-weighting easy examples appears to destabilise training on a small,
  imbalanced corpus.
- **Oversampling widened the accuracy gap** (0.082, the largest observed)
  without improving macro-F1. Resampling rare (class, tone) cells with
  replacement on a corpus this small amplifies whichever few images occupy
  them.

*Power.* With three seeds and a seed standard deviation near 0.035, this design
resolves macro-F1 differences of roughly 0.05 or larger. No observed difference
reaches that. The correct reading is *underpowered to distinguish*, not *shown
to be equivalent*.

Presented as a trade-off curve (Figure X: `rq3_tradeoff_curve.png`), the six
strategies form an overlapping cluster rather than a frontier.

---

## 5.6 RQ4 — Is the model causally sensitive to skin tone?

The audits above are correlational: they compare performance across groups that
differ in many respects simultaneously. RQ4 intervenes directly. Each image was
rendered at five tone levels by shifting only the L\* channel on skin-masked
pixels, by the offset required to move that face's ITA to a target. The a\*
channel, which carries acne erythema, was never modified, and spatial structure
was untouched, so lesion count, shape, and position are identical across the
ladder (Figure X: `rq4_shift_example.png`). Any change in prediction is
therefore attributable to tone alone.

| ΔITA | flip rate [95% CI] | Δ expected severity [95% CI] | Δ confidence |
|---|---|---|---|
| −20° (darker) | 6.8% [5.5, 8.2] | **+0.056 [+0.047, +0.066]** | −0.021 |
| −10° | 3.6% [2.6, 4.5] | **+0.019 [+0.013, +0.026]** | −0.010 |
| 0° (control) | 0.0% [0.0, 0.0] | +0.000 | +0.000 |
| +10° | 5.4% [4.2, 6.6] | −0.005 [−0.014, +0.005] | −0.022 |
| +20° (lighter) | 20.1% [17.8, 22.0] | +0.104 [+0.082, +0.126] | −0.095 |

**The answer is yes.** Two features support the claim beyond a bare
significance test. The zero-shift control returns exactly zero, confirming the
pipeline introduces no spurious variation. And the darkening arm is
**dose-responsive**: flip rate rises 3.6% → 6.8% and severity shift +0.019 →
+0.056 as the shift doubles, with both intervals excluding zero. A
dose–response relationship is materially stronger evidence than a single
significant contrast.

Magnitudes are modest in absolute terms — +0.056 is roughly 1.9% of the 0–3
severity scale — but the behaviourally relevant figure is the flip rate:
**6.8% of subjects receive a different severity grade, and therefore a
different product set, purely because their skin was rendered darker.**

### 5.6.1 The asymmetry, and what explains it

Lightening produced a substantially larger effect than darkening (20.1% against
6.8% flips). Two explanations were tested.

**Luminance clipping was rejected.** The obvious concern is that shifting L\*
upward saturates pixels and destroys texture, so the model degrades on damaged
images rather than lighter ones. Measured directly, only **0.19%** of skin
pixels clip at +20° and **0.00%** at −20°. Clipping does not account for the
asymmetry.

**Training-distribution sparsity does.** ACNE04's Method-B ITA distribution has
median 37.3°, with only **8.7%** of images above 55°. Shifting the corpus
upward therefore moves a large share of it into a thinly populated region:
33.1% of images land in the sparse tails at +20° against 21.5% at −20°, and
5.0% land outside the observed range against 0.3%. The confidence drop
corroborates it: −0.095 at +20° against −0.021 at −20°. The model is not merely
wrong at the light extreme — it is *uncertain* there, the signature of
operating beyond the density of its training data.

**This is the finding worth carrying forward.** Instability is greatest
precisely where training coverage is thinnest. That is the mechanism by which
under-representation in a corpus becomes unreliable behaviour for the
under-represented, demonstrated causally rather than inferred from an accuracy
table.

*Limitation.* The probe manipulates a rendered proxy for skin tone, not the
biological and photographic covariates that accompany tone in real subjects. It
establishes sensitivity to the tone signal as encoded in pixels; it does not
reproduce a genuinely different subject. The shift also holds b\* fixed while
solving for L\*, whereas real tone variation moves both — a further reason to
weight the ±10° and −20° results above +20°.

*Figure:* `rq4_tone_sensitivity.png`.

---

## 5.7 RQ5 — Is the review corpus representationally biased?

This arm is deliberately independent of the image pipeline. Of 1,094,411
reviews, 923,802 (84.4%) carry a usable tone band.

| band | reviews | share [95% CI] |
|---|---|---|
| Light | 728,833 | 78.9% [78.8, 79.0] |
| Medium | 168,350 | 18.2% [18.1, 18.3] |
| Deep | 26,619 | **2.9% [2.8, 2.9]** |

**The representation ratio is 27.4 : 1, Light to Deep.** The aggregate review
signal on which any review-driven recommender depends is overwhelmingly a
lighter-skin signal. This is a bias in the corpus itself, prior to any model
being trained on it.

| band | mean rating | P(recommended) | median price |
|---|---|---|---|
| Light | 4.294 [4.292, 4.297] | 83.8% | $40 |
| Medium | 4.303 [4.297, 4.308] | 84.5% | $39 |
| Deep | 4.267 [4.252, 4.280] | 84.9% | $38 |

Differences in rating and price are statistically significant — inevitably so
at this sample size — but **negligible in effect size**: ε² = 0.0001 for rating
and 0.0004 for price. Reporting effect sizes alongside p-values is what
prevents these from being oversold as evidence of differential experience.

The substantive finding is in concern coverage:

| band | mentioning PIH-family concerns | rate [95% CI] |
|---|---|---|
| Light | 20,748 | 2.85% [2.81, 2.89] |
| Medium | 7,726 | 4.59% [4.48, 4.70] |
| Deep | 1,794 | **6.74% [6.45, 7.04]** |

Reviewers with deeper skin tones raise hyperpigmentation, dark-mark, and
acne-scarring concerns at **2.4 times the rate** of lighter-toned reviewers
(χ² = 2347.5, Cramér's V = 0.050). The two findings compound: the concern most
characteristic of darker skin is raised disproportionately often by a group
contributing under 3% of the corpus. A recommender learning product-concern
associations from review text has correspondingly thin evidence precisely where
the concern is most prevalent.

*Figures:* `rq5_representation.png`, `rq5_pih_coverage.png`, `rq5_rating_ci.png`,
`rq5_price_ci.png`.

### 5.7.1 Sentiment modelling and a check on the text pipeline

A DistilBERT-base classifier was fine-tuned on 120,000 label-balanced reviews
(labels derived from star ratings; 3-star reviews dropped as ambiguous). On a
held-out set of 24,000 it achieves **accuracy 0.974, macro-F1 0.974, ROC-AUC
0.996**, with near-symmetric errors across classes.

The reason for building it, beyond discharging the methodological commitment,
is to test whether bias enters the pipeline through the *text* route as well as
the image route. If the model read one group's reviews less accurately, that
group's expressed preferences would be measured less reliably.

| tone band | n | accuracy [95% CI] |
|---|---|---|
| Light | 15,887 | 0.9751 [0.9726, 0.9773] |
| Medium | 3,630 | 0.9774 [0.9725, 0.9821] |
| Deep | 575 | 0.9704 [0.9565, 0.9827] |

**Accuracy spread 0.0070, with heavily overlapping intervals — no disparity is
demonstrated.** The text pipeline reads all three bands with effectively equal
accuracy.

The qualification is instructive rather than incidental. The Deep band
contributes only 575 test reviews, giving an interval roughly five times wider
than the Light band's. **The sample sizes reproduce the corpus imbalance
documented above:** even where a downstream model is even-handed, the data it
learns from is not, and it is that imbalance which limits what can be
established about the smallest group.

Finally, a retrospective validation. Across 2,255 products, text-derived and
rating-derived positive rates correlate at **r = 0.904**, so approximately 82%
of product-level variance is shared. **The rating proxy relied upon in §5.4 and
§5.7 was therefore a sound stand-in**, and substituting the transformer would
not have materially altered those results. The residual ~18% is where text
sentiment carries information the star rating does not.

*Note on interpretation.* Because the labels are the rating threshold,
agreement between model and rating is identical to accuracy by construction;
the complement cannot be read as evidence that the rating mislabelled those
reviews. Separating "the text disagrees with the stars" from "the model is
wrong" would require human-annotated sentiment, which this project does not
have. The product-level correlation is the finding that survives this
objection.

*Figure:* `sentiment_results.png`.

---

## 5.8 RQ6 — Cross-arm replication

**Not completed.** RQ6 requires Arm A, the reference audit on Fitzpatrick17k,
which could not be conducted because no copy of the corpus pairing its images
with its ground-truth Fitzpatrick metadata could be obtained. Sources attempted
are documented in §[limitations]. The Arm A training code is implemented and
validated but unexecuted.

The consequence is stated plainly. Following the §5.1.3 gate outcome, Arm A had
become the *primary* powered fairness audit rather than a supporting one, since
it alone offered ground-truth tone labels and a population spanning Fitzpatrick
I–VI. Its absence means **the fairness findings in this dissertation rest
entirely on a corpus with a demonstrably compressed tone range**, and the
nulls in RQ1 and RQ2 must be read in that light: they establish that no
disparity was detectable between two adjacent light-to-medium bands, not that
none exists across the full range of human skin tone.

That limitation is itself evidence for the dissertation's broader argument. The
practical inaccessibility of the field's most widely cited tone-labelled
dermatology corpus — images in one place, labels in another, complete versions
behind invitation — is a material obstacle to fairness auditing, and it is
worth reporting as such.

---

## 5.9 The deployed system

### 5.9.1 Abstention

The classifier abstains below a confidence threshold rather than emitting a
low-confidence guess. The threshold was derived from the same out-of-fold
predictions used in RQ1, as the lowest value achieving 85% selective accuracy
while retaining at least 50% coverage.

**τ = 0.85**, giving coverage 0.677 and selective accuracy 0.851 — an 8.5
percentage-point accuracy improvement over the full-coverage baseline of 0.766,
purchased by declining to advise on a third of cases.

Because abstention is itself an allocation, it was checked per group:
coverage 0.697 (Light) against 0.689 (Medium), a gap of **0.008**. Abstention
does not fall materially harder on either auditable group. Had it done so, the
system would have been quietly denying service to one group — an allocative
harm invisible to accuracy metrics.

*Figure:* `abstention_curve.png`.

### 5.9.2 Refusal behaviour

The conversational interface was tested against a purpose-built adversarial set
of 21 medical requests spanning direct, indirect, role-play, smuggled,
procedural, and emergency framings, alongside 12 benign cosmetic controls.

**Refusal rate 1.00; false-refusal rate 0.00.**

Benign controls were included deliberately: over-refusal is a genuine failure
mode, and a system that declines everything is safe and useless. Measuring both
is what makes the refusal rate meaningful.

The result required correction. The first evaluation returned a refusal rate of
**0.81**, exposing four genuine gaps — a missed plural, a reversed word order,
a phrasing outside the anticipated template, and a pronoun variant. The rules
were widened and the set re-run. That the purpose-built set located real
defects on first contact is the strongest available evidence that it constitutes
a test rather than a demonstration. Transcripts are reproduced in Appendix [X].

### 5.9.3 Architectural separation

The demonstrator runs the rules-based dialogue manager with **no language model
in the loop**. This is the strongest available form of the separation the
design requires: the interface cannot introduce a product, a claim, or a
diagnosis, because no generative component exists in the recommendation path.
Every recommendation traces deterministically from classifier, through the
documented mapping table, to the ranking function — which is the property that
makes the propagation analysis in §5.4 a measurement of the system it claims to
measure.

---

## 5.10 Summary of findings

| # | Finding |
|---|---|
| 1 | ACNE04 contains no auditable dark-skinned population; the apparent one is a lighting and inflammation artefact, robust across three ITA estimators. |
| 2 | ResNet-50 achieves 0.766 accuracy / 0.746 macro-F1 and outperforms EfficientNet-B0/B3 and ViT-Base by more than seed noise. |
| 3 | No accuracy disparity is demonstrated between Light and Medium bands, but calibration is ~30% worse for the smaller group. |
| 4 | Misclassification propagates strongly into recommendations: Jaccard overlap falls from 1.000 to 0.515. |
| 5 | That propagation is not demonstrably tone-differential on this corpus. |
| 6 | No mitigation strategy beats the unmitigated baseline; seed variance exceeds every between-strategy difference eightfold. |
| 7 | The model *is* causally tone-sensitive: 6.8% of predictions flip under a −20° ITA shift, dose-responsively, with lesion content held fixed. |
| 8 | Model instability is greatest where training tone coverage is sparsest — the causal mechanism linking under-representation to harm. |
| 9 | The review corpus is representationally biased 27:1 Light to Deep, while deeper-toned reviewers raise hyperpigmentation concerns at 2.4× the rate. |
| 10 | Refusal behaviour achieves 1.00 refusal and 0.00 false-refusal after correction of four defects located by the adversarial set. |

**The central result is the conjunction of findings 3, 5, and 7.** A model that
shows no disparity in disaggregated accuracy, and whose errors produce no
demonstrably tone-differential downstream harm, is nevertheless demonstrably
sensitive to skin tone at the mechanism level. Aggregate audits average over a
population whose tone range is narrow; a causal probe manufactures the contrast
directly and does not depend on that population existing. Disaggregated
auditing and causal probing therefore answer different questions, and reporting
only the former can license a false assurance of tone-blindness.
