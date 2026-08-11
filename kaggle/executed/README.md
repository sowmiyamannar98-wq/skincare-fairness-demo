# Executed notebooks — run records with outputs

The notebooks in the parent `kaggle/` directory are **sources**, generated from
the `.py` files by `build_notebooks.py` and carrying no outputs. The copies here
are the versions **as actually run on Kaggle's GPU**, with their cell outputs
retained, so every figure quoted in the dissertation can be traced to the run
that produced it.

## Execution status

| Notebook | Code cells | Executed | Status |
|---|---|---|---|
| `nb1_dataprep_ita_gate` | 15 | 15 | Fully executed |
| `nb1b_ita_validation` | 8 | 8 | Fully executed |
| `nb3_train_armB_acne04` | 5 | 5 | Fully executed |
| `nb4_counterfactual_tone_probe` | 9 | 8 | One cell unrun — see below |
| `nb5_mitigation_experiments` | 6 | 6 | Fully executed |
| `nb6_architecture_comparison` | 6 | 6 | Fully executed |
| `nb7_sentiment_model` | 8 | 8 | Fully executed |
| `nb2_train_armA_fitzpatrick` | — | — | **Never run — blocked** |

No notebook contains an error output.

## The two exceptions, and why they are not defects

**NB2 (Arm A, Fitzpatrick17k) was never run.** It required a copy of the corpus
pairing its images with its ground-truth Fitzpatrick metadata, and no such copy
could be obtained; six sources were attempted and are documented in the report.
The source notebook in the parent directory is implemented and hardened — it
asserts the metadata's presence and schema and refuses to proceed below a
minimum matched-image count — but it has no run record because it never ran.
This is the reason RQ6 is unanswered, and it is reported as a documented
limitation rather than an omission.

**NB4 has one unrun cell**, the final plotting cell that would have written
`rq4_tone_sensitivity.png`. Every analysis cell above it ran, so the RQ4 results
quoted in the dissertation are real outputs of this notebook. The cell was left
unrun because it indexes `summ.shift`, and `shift` collides with the pandas
`DataFrame.shift` method, so the expression returns a bound method rather than
the column. The column was subsequently renamed `ita_shift` elsewhere in the
codebase. The figure was regenerated locally from the reported summary
statistics by `scripts/10_regenerate_kaggle_figures.py`, which needs no GPU.

## Verification against the dissertation

Key numbers were checked against the values reported in the write-up:

| Notebook | Output in run record | Reported |
|---|---|---|
| NB3 | `acc=0.7660` | 0.766 |
| NB5 | ERM across seeds `0.7019 / 0.7395 / 0.7645` | 0.702 / 0.740 / 0.765 |
| NB7 | `accuracy 0.9743`, `ROC-AUC 0.9959` | 0.974, 0.996 |

## A note on running these

Kaggle mounts datasets nested at `/kaggle/input/datasets/<owner>/<slug>`, so the
`find_dir()` helpers walk the tree recursively and select the folder holding the
most images — otherwise a 13-image stub can shadow the real dataset. Always use
**Save Version → Save & Run All (Commit)** rather than an interactive Run All:
`/kaggle/working` is wiped when an interactive session ends, and outputs were
lost that way once. Each training notebook carries a `SMOKE_TEST` flag for a
short validation pass before committing a full run.
