# RQ4 — Counterfactual Tone Probe: Is the Model Causally Sensitive to Skin Tone?

_Results from `kaggle/nb4_counterfactual_tone_probe.ipynb`, run on the trained
Arm B ResNet-50 (`armB_acne_resnet50_best.pt`, out-of-fold accuracy 0.766)
across all 1,457 ACNE04 images × 5 tone levels = 7,285 forward passes._

## 1. Why a counterfactual probe

The disaggregated audit in RQ1 is correlational: it compares accuracy *across*
tone groups, and those groups differ in many ways at once — lighting, image
source, condition prevalence. A null there does not establish that the model
ignores tone, and a gap there does not establish that it responds to tone.

This probe intervenes directly. For each image it shifts the subject's skin
tone by a controlled amount and asks whether the prediction moves. Because the
intervention changes tone and nothing else, any change in output is
attributable to tone alone.

**Construction.** Working in CIELAB:

- the **a\*** channel (red–green), which carries acne erythema, is **never
  modified** — lesion appearance is preserved exactly;
- only **L\*** is shifted, and only on skin-masked pixels, by the offset
  required to move that face's ITA to the target;
- spatial structure is untouched, so lesion count, shape, and position are
  identical across the ladder.

## 2. Result — the model is causally tone-sensitive

| ΔITA | prediction flip rate | Δ expected severity [95% CI] | Δ confidence |
|---|---|---|---|
| −20 (darker) | 6.8% [5.5, 8.2] | **+0.056 [+0.047, +0.066]** | −0.021 |
| −10 | 3.6% [2.6, 4.5] | **+0.019 [+0.013, +0.026]** | −0.010 |
| 0 (control) | 0.0% [0.0, 0.0] | +0.000 [0.000, 0.000] | +0.000 |
| +10 | 5.4% [4.2, 6.6] | −0.005 [−0.014, +0.005] | −0.022 |
| +20 (lighter) | 20.1% [17.8, 22.0] | **+0.104 [+0.082, +0.126]** | −0.095 |

**The answer to RQ4 is yes.** Holding lesion content fixed, changing skin tone
alone changes the model's output. Two features of the table support this
beyond a bare significance claim:

1. **The zero-shift control is exactly zero**, confirming the pipeline
   introduces no spurious variation of its own.
2. **The darkening arm is dose-responsive** — flips rise 3.6% → 6.8% and
   severity shift rises +0.019 → +0.056 as the shift doubles, with both
   confidence intervals excluding zero. A dose–response relationship is
   considerably stronger evidence than a single significant contrast.

Magnitudes are small in absolute terms: +0.056 on a 0–3 severity scale is
roughly 1.9% of the range. The behaviourally meaningful figure is the **flip
rate** — 6.8% of subjects receive a different severity grade, and therefore a
different product set, purely because their skin was rendered darker.

## 3. The asymmetry, and what actually explains it

Lightening produced a far larger effect than darkening (20.1% vs 6.8% flips).
Two explanations were tested.

**Rejected: luminance clipping.** The obvious concern is that shifting L\*
upward saturates pixels and destroys texture, so the model degrades on damaged
images rather than lighter ones. Measured directly, this is not what happens:
only **0.19%** of skin pixels clip at +20, and **0.00%** at −20. Clipping does
not account for the asymmetry.

**Supported: training-distribution sparsity.** ACNE04's Method-B ITA
distribution has median 37.3° and only **8.7%** of images above 55°. Shifting
the corpus upward therefore moves a large share of it into a thinly populated
region:

| shift | share landing in sparse tails (<p5 or >p95) | share outside the observed range |
|---|---|---|
| −20 | 21.5% | 0.3% |
| +20 | **33.1%** | **5.0%** |

The confidence drop corroborates this: −0.095 at +20 against −0.021 at −20.
The model is not merely wrong at the light extreme, it is *uncertain* there —
the signature of operating outside the density of its training data.

**This is the finding worth reporting.** The instability is largest precisely
where training coverage is thinnest. That is the mechanism by which
under-representation in a training corpus becomes unreliable behaviour for the
under-represented, demonstrated causally rather than inferred from an accuracy
table.

## 4. How this sits alongside RQ1 and RQ2

| question | method | result |
|---|---|---|
| RQ1 | disaggregated accuracy audit | **no** resolved tone gap |
| RQ2 | propagation into recommendations | **no** resolved tone-differential propagation |
| **RQ4** | **causal counterfactual probe** | **yes — output moves with tone alone** |

Read together these are not in conflict; they are the substantive contribution.
**A model can be demonstrably tone-sensitive at the mechanism level while
showing no disparity in aggregate accuracy or downstream outcome metrics.**

Aggregate metrics average over a population whose tone range is narrow — here,
two auditable bands and no dark-skin population at all. A causal probe does not
depend on that population existing, because it manufactures the contrast
directly. The methodological implication is that disaggregated auditing and
causal probing answer different questions, and reporting only the former can
license a false assurance of tone-blindness.

## 5. Limitations

- The probe manipulates a **rendered proxy** for skin tone, not the biological
  and photographic covariates that accompany tone in real subjects. It
  establishes sensitivity to the tone signal as encoded in pixels; it does not
  reproduce a genuinely different subject.
- The ITA shift holds `b*` fixed while solving for `L*`. Real tone variation
  moves both. Wider shifts therefore drift from realistic skin loci, which is
  an additional reason to weight the ±10 and −20 results above +20.
- Effects are estimated on one architecture and one training run. The RQ3
  mitigation experiments re-estimate across strategies and seeds.

## 6. Artefacts

- `rq4_counterfactual_predictions.csv` — per-image, per-shift predictions
- `rq4_summary.csv`, `rq4_result.json` — aggregated results
- `rq4_tone_sensitivity.png` — flip rate and directional bias curves
- `rq4_shift_example.png` — visual verification that lesions are preserved
  across the tone ladder
