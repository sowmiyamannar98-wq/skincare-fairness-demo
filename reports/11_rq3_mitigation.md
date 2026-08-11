# RQ3 — Bias Mitigation Strategies and the Accuracy–Fairness Trade-off

_Results from `kaggle/nb5_mitigation_experiments.ipynb`: 6 strategies × 3 seeds
= 18 independent training runs of ResNet-50 on ACNE04 (75/25 split stratified
jointly by severity and validated Method-B tone band), ~17 min per run._

## 1. What this experiment could target

RQ1 found **no statistically resolved tone accuracy gap** on this corpus
(−0.020, 95% CI [−0.070, +0.026]), and NB1b established there is **no
auditable dark-skin population** in ACNE04 at all. There was therefore no
large tone disparity available to close, and the experiment was scoped in
advance to the two deficits the audit did find: the **calibration gap**
(Light ECE 0.120 vs Medium 0.092) and **severe-class weakness** (grade 2
F1 0.62 against 0.82 for grade 0).

## 2. Results

| strategy | macro-F1 | severe-class F1 | \|acc gap\| | \|ECE gap\| | worst-group acc |
|---|---|---|---|---|---|
| group_reweight | **0.743 ± 0.035** | 0.595 | 0.034 ± 0.021 | 0.051 ± 0.023 | **0.760** |
| erm (baseline) | 0.735 ± 0.031 | 0.591 | 0.036 ± 0.021 | 0.027 ± 0.023 | 0.756 |
| class_reweight | 0.731 ± 0.026 | 0.582 | 0.052 ± 0.013 | 0.038 ± 0.009 | 0.730 |
| oversample | 0.731 ± 0.035 | 0.570 | 0.082 ± 0.012 | 0.034 ± 0.009 | 0.723 |
| group_dro | 0.726 ± 0.011 | 0.531 | 0.033 ± 0.017 | **0.014 ± 0.016** | 0.752 |
| focal | 0.708 ± 0.036 | 0.578 | 0.070 ± 0.022 | 0.071 ± 0.051 | 0.692 |

## 3. The headline: no strategy demonstrably improves on the baseline

The best-performing strategy (`group_reweight`, macro-F1 0.743) exceeds the
unmitigated baseline (`erm`, 0.735) by **0.008** — against a seed standard
deviation of **±0.035**. The difference is roughly a quarter of the noise. On
every column, the between-strategy spread is smaller than the within-strategy
seed spread.

This is made unmistakable by the raw runs. Under `erm` alone, macro-F1 across
three seeds was **0.702 / 0.740 / 0.765** — a spread of 0.063, which is
**eight times** the largest apparent gain from any mitigation. Had this
experiment been run at a single seed, almost any ranking of the six strategies
could have been produced at will.

**The conclusion is that mitigation cannot be shown to help here, and the
reason is not that the strategies are ineffective in general — it is that
there was no resolved disparity to correct.** This follows directly from RQ1
and was anticipated when the experiment was scoped.

## 4. What can be said, carefully

- **GroupDRO produces the smallest calibration gap** (0.014 ± 0.016) and by
  far the **lowest seed variance** (±0.011 on macro-F1, a third of the others).
  Its interval still includes the baseline, so this is not a demonstrated
  improvement — but a strategy that is *more stable across seeds* is a
  meaningful practical property, and it is the one result here worth
  following up with more seeds.
- **Focal loss is the clearest loser** — lowest macro-F1 (0.708), worst
  worst-group accuracy (0.692), and the largest and most volatile calibration
  gap (0.071 ± 0.051). Down-weighting easy examples appears to destabilise a
  model already trained on a small, imbalanced corpus.
- **Oversampling widened the accuracy gap** (0.082 ± 0.012, the largest of
  any strategy) while not improving macro-F1. Resampling rare (class, tone)
  cells with replacement on a corpus this small evidently amplifies whichever
  few images occupy those cells.
- **No strategy improved severe-class F1** beyond the baseline's 0.591;
  group_reweight's 0.595 is well inside noise. The severe grades remain the
  model's weakest point, and the constraint is the 186 and 137 available
  images, not the loss function.

## 5. Power

With 3 seeds and a seed standard deviation near 0.035, this design can only
resolve macro-F1 differences of roughly **0.05 or larger**. Differences below
that are indistinguishable from run-to-run variation, and none of the observed
differences reach it. The correct reading is *underpowered to distinguish*,
not *shown to be equivalent*. Increasing seeds is the direct remedy and is the
recommended extension; the GroupDRO stability result is the most promising
candidate for it.

## 6. Interpretation for the dissertation

Reported as a trade-off curve rather than a single-point claim
(`rq3_tradeoff_curve.png`), the picture is a cluster, not a frontier: the six
strategies occupy overlapping positions whose error bars intersect. This is
itself the finding, and it carries a methodological point worth stating
plainly —

**Mitigation results reported at a single seed are not trustworthy.** The
seed-to-seed spread on this corpus exceeds every between-strategy difference.
A study reporting "strategy X reduced disparity by 0.03" from one run would be
reporting noise, and would be indistinguishable from a study reporting the
opposite. This is the same argument the pre-committed protocol (§7.2) exists
to enforce, arriving here as empirical evidence rather than as a principle.

## 7. Artefacts

- `rq3_mitigation_raw.csv` — all 18 runs, per seed
- `rq3_mitigation_summary.csv` — aggregated with standard deviations
- `rq3_tradeoff_curve.png` — macro-F1 against accuracy gap and calibration gap
- `rq3_result.json` — machine-readable summary
