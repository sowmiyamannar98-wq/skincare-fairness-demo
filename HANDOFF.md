# PROJECT HANDOFF — READ THIS FIRST

**Purpose.** A complete, self-contained state dump so a fresh assistant (or a
future you) can continue this dissertation without re-deriving anything.
**Last updated: 10 August 2026.**

## ⏰ URGENT CONTEXT

**Submission is in ~1 day.** State:

1. **CODING IS COMPLETE.** Do not start new analysis.
2. **THE REPORT IS WRITTEN.** Every section, in the university template, built
   by a script. See **§9b** — that is the most important section for a new
   session. Output: `report_build/MSc_Project_Report_Skin_Fairness.docx`
   (22 figures, 19 tables, 61 IEEE references, no TODOs).
3. **THE APP IS REBUILT AND DEPLOYED.** See **§11**.
4. RQ6 remains blocked and is written up as a documented limitation, which
   turns the gap into an argument rather than a hole.

### ⚠️ WHAT IS ACTUALLY OUTSTANDING — start here

| # | Item | Who | Notes |
|---|---|---|---|
| 1 | **Surname missing.** `report_build/report_content.py` has `AUTHOR = "Sowmiya"` | needs student | Fills title page + declaration |
| 2 | **Word count is 120% of budget** (Evaluation 145%) | needs a decision | If limits are enforced, Evaluation must lose ~900 words. See §9b |
| 3 | **Verify Streamlit deployment** | needs student | App showed a placeholder page on last check; get the build log. See §11 |
| 4 | **3 references unverified** | needs checking | `wang2022`, `bower2017`, `groh2022` were written from memory — confirm against originals |
| 5 | Screencast (Appendix D) + artefact URL into Appendix C | needs student | Appendix C currently points at the code folder only |

### Paste this to start a new session

> I'm continuing an MSc dissertation, submission ~1 day away. Read
> `D:\sowmiya_FYP\HANDOFF.md` first — especially §9b (the report build
> pipeline) and §11 (the app and its deployment). The report is fully written
> and builds from `report_build/`. Do not start new analysis. The outstanding
> items are listed at the top of HANDOFF.

---

## 1. What this project is

**MSc Data Science dissertation:** *Does Bias Propagate? Fairness in Automated
Acne Severity Assessment and Downstream Skincare Recommendation.*

Proposal: `MSc_Project_Proposal_Skin_Fairness.md` — the contract this codebase
delivers against.

**Research question:** do skin-tone disparities in an automated acne-severity
classifier *propagate* into the skincare recommendations that depend on them?

**Contribution is application and evidence**, not a new fairness method.

```
Arm A:  Fitzpatrick17k ──► fairness audit ONLY   (BLOCKED — see §6)
Arm B:  ACNE04 ──► ResNet-50 ──► [mapping table] ──► recommender ──► Sephora
RQ5:    Sephora reviews ──► independent representational audit
RQ6:    compare Arm A vs Arm B                    (BLOCKED — needs Arm A)
```

The three datasets **do not merge on any key**. Arm B connects to Sephora
through the severity→ingredient mapping table (`reports/05_mapping_table.md`).
Arm A is deliberately never connected to the recommender (proposal §5.3).

### Datasets

| Role | Dataset | Kaggle slug |
|---|---|---|
| Arm B images | ACNE04, 1,457 images | `lexuanhieu131297/acne-severity-classification` |
| Recommender + RQ5 | Sephora, ~1.09M reviews | `nadyinky/sephora-products-and-skincare-reviews` |
| Arm A | Fitzpatrick17k | **no usable mirror — see §6** |

ACNE04 images sit at `.../Classification/JPEGImages/`, named `levle{0-3}_*.jpg`
where the digit is the Hayashi grade. **Kaggle mounts nested** at
`/kaggle/input/datasets/<owner>/<slug>` — notebooks walk the tree recursively.

---

## 2. Status

| RQ | Question | Status | Report |
|---|---|---|---|
| RQ1 | Equitable across tone? | ✅ | `reports/04` |
| RQ2 | Does bias propagate? | ✅ | `reports/07` |
| RQ3 | Which mitigation works? | ✅ | `reports/11` |
| RQ4 | Causally tone-sensitive? | ✅ | `reports/10` |
| RQ5 | Is the corpus biased? | ✅ | `reports/03` |
| RQ6 | Cross-arm replication | ❌ blocked | — |

**Also complete:** trained model, ITA validation, architecture comparison,
mapping table, recommender, demonstrator app, adversarial safety testing,
derived abstention threshold, sentiment model, **the full report (§9b)**, and
**the rebuilt + deployed app (§11)**.

**Outstanding:** Arm A + RQ6 (blocked); optional stricter relevance metric for
RQ2. Everything else outstanding is in the table at the top of this file.

---

## 3. EVERY RESULT, WITH NUMBERS

### Trained model
ResNet-50 (ImageNet-pretrained, `timm`), 4-class Hayashi severity, repeated
5-fold CV over all 1,457 images:
- **accuracy 0.766, macro-F1 0.746**, stable across folds (0.723–0.774)
- per-class F1: 0.82 / 0.77 / **0.62** / 0.78 (grade 2 weakest — 186 imgs)
- checkpoint: `data/trained weights/armB_acne_resnet50_best.pt` (94 MB)

### Architecture comparison (`reports/12`)
4 architectures × 3 seeds:

| architecture | params | macro-F1 | ECE |
|---|---|---|---|
| **resnet50** | 23.5M | **0.731 ± 0.026** | 0.108 |
| efficientnet_b0 | 4.0M | 0.690 ± 0.039 | 0.190 |
| efficientnet_b3 | 10.7M | 0.679 ± 0.030 | 0.183 |
| vit_base | 85.8M | 0.639 ± 0.038 | **0.085** |

- **RESOLVED = YES** (spread 0.092 vs pooled seed sd 0.033).
- Capacity does not help; ViT lacks CNN inductive biases, needs more data.
- **Calibration runs opposite to accuracy** — ViT best-calibrated despite worst
  accuracy. Matters for the abstention threshold.
- Reproducibility check: NB6 resnet50 (0.731) ≈ NB5 erm (0.735).
- **Confound flagged:** all trained at LR 3e-4 (tuned for ResNet); ViT normally
  uses 1e-4–5e-5, so part of its deficit is hyperparameter mismatch.

### RQ1 — fairness audit (`reports/04`)
- Light (n=538): acc 0.762 [0.723, 0.797], **ECE 0.120**
- Medium (n=724): acc 0.782 [0.751, 0.812], **ECE 0.092**
- **Accuracy gap −0.020, 95% CI [−0.070, +0.026] → NOT resolved**
- Calibration gap is the real finding: more confidently wrong on the smaller
  group.
- Deep band **excluded** — see §4.

### RQ2 — propagation (`reports/07`) — PRINCIPAL CONTRIBUTION
Each subject run through the *same deterministic* recommender twice: on true
grade vs predicted grade. All divergence attributable to the classifier.
- correct: Jaccard **1.000** · misclassified: Jaccard **0.515**
- **penalty +0.485 [0.465, 0.504]** — propagation is real and large
- **NOT tone-differential:** Jaccard gap −0.010 [−0.035, +0.015]; relevance
  −0.007 (resolved but below 0.05 materiality); price −0.29 USD unresolved
- Missed referrals: Light 2.2%, Medium 4.0%, gap unresolved
- Catalogue coverage only ~6% (~100 of 1,701 products ever surfaced)
- Pipeline P@10 0.993 vs popularity baseline 0.425
- **Known weakness:** precision@k saturated (~0.99) — Jaccard is the
  informative metric

### RQ3 — mitigation (`reports/11`)
6 strategies × 3 seeds:

| strategy | macro-F1 |
|---|---|
| group_reweight | 0.743 ± 0.035 |
| erm (baseline) | 0.735 ± 0.031 |
| class_reweight | 0.731 ± 0.026 |
| oversample | 0.731 ± 0.035 |
| group_dro | 0.726 ± 0.011 |
| focal | 0.708 ± 0.036 |

- **NULL: no strategy beats baseline** (best gain +0.008 vs seed sd ±0.035)
- **Killer evidence:** `erm` alone across seeds gave 0.702 / 0.740 / 0.765 —
  spread 0.063, **8× the largest apparent gain**. Single-seed mitigation
  results are noise.
- GroupDRO: smallest calibration gap (0.014), lowest seed variance (±0.011)
- Focal loss clearly worst; oversampling widened the accuracy gap (0.082)
- Underpowered below ~0.05. Say "underpowered to distinguish", NOT "equivalent"

### RQ4 — counterfactual tone probe (`reports/10`) — MOST INTERESTING

| ΔITA | flip rate | Δ expected severity |
|---|---|---|
| −20 | 6.8% | **+0.056 [+0.047, +0.066]** |
| −10 | 3.6% | **+0.019 [+0.013, +0.026]** |
| 0 (control) | 0.0% | 0.000 |
| +10 | 5.4% | −0.005 (null) |
| +20 | 20.1% | +0.104 [+0.082, +0.126] |

- **YES — causally tone-sensitive.** Zero-shift control exactly zero; darkening
  arm **dose-responsive** (strongest evidence form).
- +20 asymmetry: **clipping REJECTED** (0.19% pixels clip at +20, 0.00% at −20).
  Cause is **training-distribution sparsity** — only 8.7% of ACNE04 above ITA
  55°; +20 pushes 33% into sparse tails vs 21.5% at −20; confidence collapses
  (−0.095 vs −0.021).
- **Finding: model least stable exactly where training coverage is thinnest** —
  causal mechanism linking under-representation to harm.

### RQ5 — review corpus audit (`reports/03`)
- Representation **27.4:1 Light:Deep** (728,833 vs 26,619 reviews)
- Rating/price by tone: significant but **negligible effect size**
  (ε² 0.0001–0.0004)
- PIH mentions: **2.85% Light → 4.59% Medium → 6.74% Deep** (2.4×), while Deep
  contributes <3% of reviews (χ²=2347.5, Cramér's V=0.050)

### Sentiment model (`reports/13`) — NB7, RUN
DistilBERT-base on 120k label-balanced reviews, held-out 24k:
- **accuracy 0.9743, macro-F1 0.9743, ROC-AUC 0.9959**
- By tone: Light 0.9751 [0.9726,0.9773] n=15,887 · Medium 0.9774 [0.9725,0.9821]
  n=3,630 · Deep 0.9704 [0.9565,0.9827] n=575
- **Spread 0.0070 → NO disparity demonstrated.** Deep n=575 gives an interval 5×
  wider — sample sizes reproduce the RQ5 corpus imbalance.
- **Product-level correlation with rating proxy: r = 0.904** (2,255 products)
  → the rating proxy used throughout earlier analysis was a SOUND stand-in;
  substituting the transformer would not materially change RQ2/RQ5. This is a
  validation result, not a corrective one.
- ⚠️ **CORRECTION — do not repeat the notebook's own framing.** NB7 prints
  "agreement with rating 0.9743 / disagreement 2.6% = share the proxy got
  wrong". **That is invalid**: labels ARE the rating threshold, so agreement ==
  accuracy by construction. The 2.6% is model error, not evidence the rating
  mislabelled anything. Only the r=0.904 product-level correlation survives.
  `reports/13` states this correctly.

### Demonstrator + safety
- **Abstention τ = 0.85**, *derived* (`reports/08`): coverage 0.677, selective
  accuracy 0.851, per-group coverage gap only 0.008
- Adversarial testing (`reports/09`): **refusal 1.00, false-refusal 0.00** over
  21 adversarial + 12 benign controls
  - *First run scored 0.81 and caught 4 real bugs* (missed plural
    "antibiotics"; reversed order "lips and throat are swelling"; non-"how do
    I" phrasing; "what condition is it" vs "is this"). Good appendix material.
- Ingredient parsing: only **473 products (19.5%)** carry any acne active vs
  **1,339 (55.3%)** anti-ageing/hydration — corroborates Lee et al. (2024)

---

## 4. THE PIVOTAL FINDING — read before touching tone analysis

The first ITA implementation labelled ACNE04 **59.5% "Deep"**. ACNE04 is
predominantly **Chinese** — not credible.

Validated by rendering **montages of real faces grouped by estimated tone**
(`kaggle/nb1b_ita_validation.py`). The "Deep" images were light-skinned faces
in **shadow** or reddened by **acne inflammation**.

| method | median ITA | Light/Medium/Deep | verdict |
|---|---|---|---|
| A (original) | 14.8° | 4 / 36 / 59% | ❌ broken |
| **B (adopted)** | **37.3°** | 37 / 50 / 13% | ✅ = Fitzpatrick III |
| C (central crop) | 45.8° | 67 / 26 / 7% | sensitivity variant |

**Method B is the primary tone estimate everywhere** — drops the reddest 25% of
skin pixels (acne) and keeps the brighter half (well-lit skin). B/C band
agreement is 60.1%.

**CRITICAL: under EVERY method, ACNE04 contains no genuinely dark-skinned
subjects.** The residual "Deep" is artifact.

**Consequences:** §10.1 gate outcome is **NO-GO** → Arm A becomes the primary
audit; Arm B becomes under-powered replication; all tone analysis restricted to
**Light vs Medium**; compressed tone range **reported as a finding**.

---

## 5. THE SYNTHESIS — the dissertation's central argument

| RQ | Method | Result |
|---|---|---|
| RQ1 | disaggregated accuracy audit | **no** resolved tone gap |
| RQ2 | propagation into recommendations | **no** resolved tone-differential propagation |
| RQ4 | causal counterfactual probe | **YES — output moves with tone alone** |

> **A model can be demonstrably tone-sensitive at the mechanism level while
> showing no disparity in aggregate accuracy or downstream outcome metrics.**

Aggregate audits average over a population whose tone range is narrow (two
auditable bands, no dark-skin population). A causal probe manufactures the
contrast directly and does not depend on that population existing.
**Disaggregated auditing and causal probing answer different questions, and
reporting only the former can license a false assurance of tone-blindness.**

Secondary point (RQ3 + NB6): **seed variance must be characterised before any
difference is claimed.** The same noise-floor test returns *unresolved* for
mitigation and *resolved* for architecture — evidence the test discriminates.

### Weaknesses to disclose, not hide
1. **RQ2 precision@k saturated** (~0.99 both arms) — mapping table endorses
   overlapping classes at adjacent grades. **Jaccard is the informative
   metric.** Stricter grade-specific reference standard is the fix.
2. **RQ2 attribute profiles are simulated** (Sephora distribution, seed 42).
3. **ITA is a proxy**, not ground truth.
4. **RQ4 manipulates rendered tone**, not real subjects.
5. **Architecture comparison shares one LR** across all models.
6. **All fairness findings rest on a corpus with compressed tone range.**

---

## 6. THE BLOCKER — Fitzpatrick17k

Arm A (and RQ6) needs Fitzpatrick17k **with its metadata CSV** (`md5hash`,
`fitzpatrick_scale`/`fitzpatrick`, and a condition label like
`three_partition_label`).

**Tried and failed — do not repeat:**

| source | outcome |
|---|---|
| `mobaswiralfarabi/fitzpatrick17k-original` | 16,574 images, **zero labels** |
| `mobaswiralfarabi/fitzpatrick17k` | 13 images (stub) |
| `nazmussadat013/fitzpatrick-17k-dataset` | 404, deleted |
| `asosenge/fitzpatrick17k` | never successfully attached |
| BTTAI `bttai-ajl-2025` | **invitation required** |
| `github.com/mattgroh/fitzpatrick17k` | CSV of *URLs*, many dead |

`kaggle/nb2_train_armA_fitzpatrick.py` is written and hardened (asserts CSV
exists, asserts columns, reports join retrieval %, fallback join, refuses
<200 matched images). Attach a labelled dataset and it should run.

**With 2 days left this is realistically dead.** Draft supervisor email at
`reports/supervisor_update_email.md`. **Write it up as a documented limitation
(see `DRAFT_results_chapter.md` §5.8) — that section turns the gap into an
argument about the practical inaccessibility of tone-labelled corpora.**

---

## 7. File inventory

### Local scripts — all run
| file | produces |
|---|---|
| `scripts/01_eda_sephora.py` | `reports/01` |
| `scripts/02_clean_and_build.py` | `data/processed/*.parquet` |
| `scripts/03_rq5_review_audit.py` | `reports/03` — RQ5 |
| `scripts/04_armB_fairness_audit.py` | `reports/04` — RQ1 |
| `scripts/05_ingredients_and_mapping.py` | `reports/05` — mapping table |
| `scripts/recommender.py` | deterministic ranker (module) |
| `scripts/07_propagation_analysis.py` | `reports/07` — **RQ2** |
| `scripts/08_abstention_threshold.py` | `reports/08`, `app/abstention_config.json` |
| `scripts/09_adversarial_test.py` | `reports/09` — safety appendix |
| `scripts/10_regenerate_kaggle_figures.py` | rebuilds 4 Kaggle figures locally from reported stats |

### Kaggle notebooks
`.py` = editable source; run `python kaggle/build_notebooks.py` to regenerate
`.ipynb` for upload.

| notebook | status | dataset |
|---|---|---|
| `nb1_dataprep_ita_gate` | ✅ run | all three |
| `nb1b_ita_validation` | ✅ run | ACNE04 |
| `nb2_train_armA_fitzpatrick` | ❌ blocked | labelled Fitzpatrick |
| `nb3_train_armB_acne04` | ✅ run | ACNE04 |
| `nb4_counterfactual_tone_probe` | ✅ run | ACNE04 + checkpoint |
| `nb5_mitigation_experiments` | ✅ run | ACNE04 |
| `nb6_architecture_comparison` | ✅ run | ACNE04 |
| `nb7_sentiment_model` | ✅ run | Sephora |

### Reports (all generated, all real)
`01_sephora_eda` · `02_build_summary` · `03_rq5_review_audit` ·
`04_armB_fairness_audit` · `05_mapping_table` · `07_rq2_propagation` ·
`08_abstention_threshold` · `09_adversarial_transcripts` ·
`10_rq4_counterfactual` · `11_rq3_mitigation` · `12_architecture_comparison` ·
`13_sentiment_model` · `supervisor_update_email` · **`DRAFT_results_chapter.md`**

### Demonstrator — **see §11 for the rebuilt UI and deployment**
`app/inference.py` · `app/dialogue.py` · `app/safety.py` · `app/app.py` ·
`app/app_basic.py` (original minimal UI) · `app/abstention_config.json`

Deployment copies: `streamlit_cloud/` (live) and `hf_space/` (unused).

Run: `streamlit run app/app.py`

**Architecture constraint (§7.7):** this build has **no language model in the
loop**. The interface cannot invent a product or diagnosis because no
generative component exists in the path. **Do not add an LLM to the
recommendation path** — it would invalidate the RQ2 propagation analysis.

### Figures — 18 in `reports/figures/`, only 2 still missing

Four Kaggle figures were **rebuilt locally** by
`scripts/10_regenerate_kaggle_figures.py` from the summary statistics in
reports/10-13 (no GPU, no Kaggle needed): `rq4_tone_sensitivity.png`,
`rq3_tradeoff_curve.png`, `arch_comparison.png`, `sentiment_by_tone.png`.
They render at publication quality. Re-run that script any time to reproduce.

**ALL FIGURES ARE NOW PRESENT — 22 files, nothing outstanding.** The four ITA
figures (`ita_montage_method_A/B/C.png`, `ita_method_distributions.png`) were
downloaded from NB1b and copied in. The only figure never retrieved is
`rq4_shift_example.png`, which is optional and not referenced as essential.

### Data
- `data/sephora/archive/` — raw CSVs (~500 MB, git-ignored)
- `data/processed/` — built parquet layer
- `data/kaggle_out/` — downloaded Kaggle outputs (analysis scripts read here)
- `data/trained weights/` — **the checkpoint, keep safe**

---

## 8. Environment and gotchas

Python 3.11.9, Windows, `D:\sowmiya_FYP`. Installed: pandas 3.0, numpy 2.4,
scipy, sklearn 1.8, torch 2.10 (CPU), streamlit 1.55, timm 1.0.28, opencv 5.0.

Kaggle for GPU. **Always "Save Version → Save & Run All" (commit)**, never
interactive Run All — `/kaggle/working` is wiped when an interactive session
ends (outputs were lost this way once). Every training notebook has a
`SMOKE_TEST` flag: `True` for a ~4 min check, then `False` for the real run.

### Gotchas learned the hard way
1. **Kaggle mounts nested**: `/kaggle/input/datasets/<owner>/<slug>`. All
   `find_dir()` helpers walk recursively **and pick the folder with the most
   images** (so a 13-image stub can't shadow the real dataset).
2. **Never name a pandas column `shift`** — collides with `DataFrame.shift`;
   `df.shift == 0` silently returns `False`, then `df[False]` raises
   `KeyError: False`. Cost an hour. Column is now `ita_shift`.
3. Windows console is cp1252 — wrap `print()` in try/except UnicodeEncodeError
   when writing UTF-8 reports (all scripts do).
4. Merging frames that both have `tone_band` silently suffixes the column.
   `scripts/04` drops the stale one before joining.
5. **The Kaggle API token on this machine belongs to account `Tharun_2431`,
   which does NOT own the FYP notebooks.** `kaggle kernels output` therefore
   cannot fetch them. The notebooks live on the student's own Kaggle account.
   Options: download via the Kaggle web UI, or have her set those notebooks
   Public so the existing token can reach them. Do not ask anyone to hand
   over credentials.

---

## 9. What to do next

**CODING IS FINISHED. THE REPORT IS WRITTEN.** Nothing is queued. The remaining
work is the five items in the table at the top of this file.

`reports/DRAFT_results_chapter.md` is now **superseded** — it was the source
material, and its content has been split into Implementation and Evaluation in
`report_build/report_content.py`. Keep it for reference; do not edit it
expecting changes to reach the report.

Still worth doing if time allows: spot-check ~5 numbers against their source
reports, and send `reports/supervisor_update_email.md` if not already sent —
it documents that Arm A was pursued.

### If asked to do more analysis
Push back. Marginal value is near zero and the deadline is the binding
constraint. The one genuinely open refinement (a stricter grade-specific
relevance metric for RQ2, section 5 weakness 1) is already disclosed as a
limitation in the report, which is the correct treatment at this stage.

## 9b. THE REPORT — WRITTEN. How to rebuild and edit it.

**The report is complete.** All 26 sections, no TODOs, no placeholders.

### Build it

```
cd D:\sowmiya_FYP\report_build
python build_report.py
```

Output: `report_build/MSc_Project_Report_Skin_Fairness.docx`
Prints: `inserted 26/26 sections; 22 figures, 19 tables, 61 references`

### The two files

| File | Role |
|---|---|
| `report_build/report_content.py` | **All prose. This is the file you edit.** |
| `report_build/build_report.py` | Injects it into the template. Rarely needs touching. |

Template source: `D:\Final year project\Report\MSc Project Report Template.docx`
(read-only — the build never modifies it). Figures come from `reports/figures/`.

### How report_content.py works — read before editing

- Each template heading maps to a Python list of strings, one per paragraph:
  `CH1_INTRO`, `CH1_PROBLEM`, … `CH4_EVAL`, `CH5_REFLECT`, `APPENDICES`.
- **A string like `"3.4 Ingredient parsing"` becomes a Heading2.** Any string
  matching `^\d\.\d [A-Z]` and under 90 chars is auto-promoted.
- **Figures:** `"[[FIG:filename.png|Caption text]]"` inserts the image plus a
  numbered caption with a real SEQ field, so the List of Figures populates.
- **Tables:** a tuple `('TABLE', 'Caption', [[row], [row], ...])`. First row is
  the header. Tables and captions are excluded from the word count — **pushing
  detail into tables is the main lever for meeting budgets.**
- **Citations:** write `[key]` or `[key1, key2]` inline, where key is a key of
  the `CITATIONS` dict. The builder resolves them to IEEE numbers in order of
  first appearance and emits the reference list in that order. **Never
  hand-number anything.** Uncited keys are dropped and printed as a warning.

### Current word counts — ⚠️ THE OPEN ISSUE

| Section | Words | Budget | |
|---|---|---|---|
| Abstract | 524 | 500 | 105% |
| Introduction | 1,356 | 1,000 | 136% |
| Literature – Technology Review | 1,812 | 1,500 | 121% |
| Implementation | 3,172 | 3,000 | 106% |
| **Evaluation + Related Works** | **2,899** | **2,000** | **145%** |
| Conclusion | 1,631 | 1,500 | 109% |
| **TOTAL** | **11,394** | **9,500** | **120%** |

If per-section limits are enforced, Evaluation must lose ~900 words. Two levers,
in order of preference: convert dense prose into `('TABLE', ...)` tuples (free —
tables are excluded from the count), then cut interpretation. Do **not** cut
§4.11 Synthesis — that is the dissertation's central argument.

A word-count checker is in the session history; the quick version is to strip
`[[FIG:...]]` markers and `[citation]` keys, then count whitespace tokens over
each `CHx_*` list, skipping strings that match the Heading2 pattern.

### Style rules the template enforces
- **Third-person observational voice only** — no "I", "we", "my", "our", "us".
  Current text is clean; re-check after editing. Note `\bI\b` false-positives
  from "Fitzpatrick I to VI".
- 12pt Arial or Calibri, 1.5 line spacing, justified, >=2 cm margins
- Every figure and table captioned and numbered — handled by the builder
- IEEE referencing — handled by the builder
- Each section starts on a new page; all text black
- **Word count excludes figures, tables, captions, references, appendices**

### After building, in Word
Select the **List of Figures** and **List of Tables** and press **F9** to
populate them. They are real TOC fields and stay empty until refreshed.

### References
61 entries in `CITATIONS`. **Three were reconstructed from memory and are
unverified: `wang2022`, `bower2017`, `groh2022`.** Check them against the
originals before submission. `lee2024` was verified against PubMed and is
correct (J. Cosmetic Dermatology, vol. 23, no. 6, pp. 2066-2077, 2024).

### Where the content came from
`reports/DRAFT_results_chapter.md` was the source and is now **superseded**. Its
~4,950 words were split: method and pipeline description into **Implementation**
(Ch3), findings into **Evaluation** (Ch4), per `reports/REPORT_PLAN.md` §2.
Chapters 1, 2 and 5, the appendices and the reference list were written fresh.
Do not edit the draft chapter expecting changes to reach the report.

## 10. How to work on this

- **Report nulls honestly.** RQ1, RQ2, RQ3 all returned nulls. Proposal §8.3
  explicitly says a rigorous null with intervals demonstrating power is a valid
  outcome. **Do not manufacture findings.**
- **Separate statistical resolution from practical materiality.** With ~1M
  reviews or ~1,200 users, trivial differences reach significance.
  `scripts/07` implements explicit materiality thresholds.
- **Effect sizes over p-values** — ε², Cramér's V, bootstrapped CIs throughout.
- **The recommender must stay deterministic** — that is what makes the
  propagation measurement meaningful.
- **Flag weaknesses before an examiner finds them.** Prior sessions corrected
  their own errors plainly when data contradicted a hypothesis (the ITA
  over-darkening; the rejected clipping explanation in RQ4). That posture is
  part of why the work holds up.
- **With 2 days left: do not start new analysis.** Help finish the write-up.

---

## 11. THE APP — rebuilt UI, and deployment

### The architectural constraint still holds — do not break it

`app/app.py` was rebuilt with a chat-style interface (message thread,
quick-reply buttons, progress rail in the sidebar, product cards, styled
refusal banners, theme-aware CSS). **Only the presentation layer changed.**
`dialogue.py`, `safety.py`, `inference.py` and `scripts/recommender.py` keep
their audited logic.

Quick-reply buttons are a keyboard shortcut, not a second code path: each
submits ordinary text through the same `DialogueManager.handle()` parser a typed
reply uses. Verified — "Oily", "Yes", "No", "Under 50" all parse correctly.

**There is still no language model anywhere in the app.** Do not add one. It
would invalidate the RQ2 propagation analysis, and the report claims this
property in Ch3 §3.9, Ch4 §4.10 and the Space README.

`app/app_basic.py` preserves the original minimal interface over the identical
backend, in case the new one ever needs to be compared against it.

### Memory — matters on free hosting

Measured peak: **~478 MB app + ~150 MB Streamlit server ≈ 628 MB**, against a
~1 GB free-tier limit (61%). Flat across repeated predictions, so no leak.

Two optimisations are in `inference.py` and should not be reverted casually:
- `torch.load(..., mmap=True)` memory-maps the checkpoint instead of reading a
  second full copy. Saves ~70 MB. Falls back automatically on older torch.
  **Verified to give byte-identical predictions.**
- `torch.set_num_threads(1)` — extra threads buy nothing on one 224x224 forward
  pass but each carries its own memory arena.

The model loads lazily: sessions where nobody uploads a photo stay near 320 MB.
`.streamlit/config.toml` caps uploads at 10 MB.

### Deployment folders

Two self-contained copies exist. **They are generated from `app/` and
`scripts/` — edit the originals and re-copy, don't edit these directly.**

| Folder | Target | State |
|---|---|---|
| `streamlit_cloud/` | Streamlit Community Cloud | **pushed and deployed** |
| `hf_space/` | Hugging Face Spaces | committed locally, never successfully deployed |

**Live app:** https://skincare-fairness-demo.streamlit.app/
**GitHub:** https://github.com/sowmiyamannar98-wq/skincare-fairness-demo

⚠️ **Deployment is UNVERIFIED.** On the last check the URL still served
Streamlit's placeholder page rather than the app. Either it was still building
(torch is a slow install) or the build failed. **Check the build log via
"Manage app" on the app page before citing this URL in Appendix C.**

Most likely failure if it did fail: `requirements.txt` pins
`torch==2.10.0+cpu`, matched to the version that wrote the checkpoint. If that
exact wheel is unavailable for Linux/cp311, relax to unpinned `torch` — the
`+cpu` local version sorts above the plain PyPI release, so pip should still
take the small CPU build rather than the multi-gigabyte CUDA one. Then
`git push`; Streamlit redeploys automatically.

### Why Hugging Face was abandoned

Four separate free-tier walls, recorded so nobody retries it:
1. Streamlit is no longer offered in the Space creation SDK picker
2. Docker SDK is gated behind a paid plan
3. Gradio Spaces on the free tier are **forced onto ZeroGPU**, and ZeroGPU only
   works with Gradio — so a Streamlit Space cannot get free CPU hardware
4. The account had already hit the 2-ZeroGPU-Space limit

`hf_space/` is complete and correct if HF's policy ever changes. The README
frontmatter (`sdk: streamlit`, `app_file: app/app.py`) is what determines the
runtime, regardless of which SDK the creation form forces you to pick.

### Licensing note, unresolved
Both deployment folders bundle `product_ingredient_flags.parquet` (213 KB) — a
derived table of product metadata and parsed ingredient flags, required for the
ranker to run. The source Sephora dataset's Kaggle licence **was never
confirmed** (the page is JS-rendered and could not be read). Confirm it before
leaving the GitHub repo public. No raw source dataset is redistributed.

### Privacy
The app processes uploads in memory and retains nothing. The deployed Space is
public, so it is a public face-upload tool for a model documented as
tone-sensitive and unvalidated on dark skin. Setting the repo/app private is a
defensible call; the README documents the limitation either way.
