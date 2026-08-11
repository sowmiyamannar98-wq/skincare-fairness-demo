# Mapping the work into the MSc Project Report Template

## 1. The problem to solve first: word budgets

The template imposes per-section limits totalling ~10,000 words. The material
already written does not fit as-is, and the mismatch is not small.

| Template section | Budget | What exists now |
|---|---|---|
| Abstract | 500 | not written |
| Introduction | 1,000 | not written |
| Literature – Technology Review | 1,500 | not written |
| **Implementation** | **3,000** | *(part of the 4,950-word results draft)* |
| **Evaluation and Results** | **2,000** | `DRAFT_results_chapter.md` = **4,950 words** |
| Conclusion | 1,500 | not written |

**The results draft is 2.5× the Evaluation budget.** It must be *split*, not
trimmed: roughly 60% of it is method/pipeline description that belongs in
**Implementation**, and only the findings belong in **Evaluation and Results**.

## 2. The split

### → Implementation (3,000 words)
Take from `DRAFT_results_chapter.md`:

| Source | What to carry over |
|---|---|
| §5.1.1–5.1.2 | Corpus characterisation; ITA derivation; the three estimators and why Method B was adopted |
| §5.1.4 | Sephora corpus, ingredient parsing, controlled vocabulary |
| `reports/05_mapping_table.md` | The severity→ingredient mapping table — this is a design artefact and belongs in Implementation |
| §5.2 (protocol only) | Training setup, repeated stratified CV, class weighting |
| §5.4.1 | The two-arm counterfactual design for RQ2 |
| §5.6 (method only) | How the tone shift is constructed in CIELAB |
| §5.9.3 | Architectural separation of the demonstrator |
| `scripts/recommender.py` | The deterministic ranker — scoring components |

**Framing:** Implementation answers *how the system was built*. Keep numbers out
of it except where a design decision depended on one.

### → Evaluation and Results (2,000 words)
Take from `DRAFT_results_chapter.md`:

| Source | What to carry over |
|---|---|
| §5.1.3 | The go/no-go gate outcome — the compressed tone range finding |
| §5.2.1 | Architecture comparison table + the resolved/unresolved verdict |
| §5.3 | RQ1 — accuracy null, calibration gap |
| §5.4.2–5.4.3 | RQ2 — the 0.515 Jaccard result and the tone-differential null |
| §5.5 | RQ3 — mitigation null and the seed-variance argument |
| §5.6 | RQ4 — the causal probe table and the sparsity explanation |
| §5.7 | RQ5 — the 27:1 representation finding |
| §5.7.1 | Sentiment model, disaggregated |
| §5.9.1–5.9.2 | Abstention threshold; adversarial refusal results |
| §5.10 | **The synthesis — this is the most important paragraph in the report** |
| — | **Related Works** subsection (required by the template, currently missing) |

**This is tight.** 2,000 words for six research questions means roughly 300
words each. Lead every subsection with the number, then one sentence of
interpretation. Push detail into tables and figures, which do not count toward
the word limit.

### → Introduction (1,000 words)
Draw from `MSc_Project_Proposal_Skin_Fairness.md` §1 (Motivation) and §3 (Aim
and RQs). Required subsections: Problem Description, Objectives, Methodology,
**Legal/Social/Ethical/Professional**, Background, Structure of Report.

The ethics subsection is straightforward — the proposal §11 already covers it:
cosmetic tool not a medical device, abstention below threshold, refusal of
medical requests, public pre-existing datasets, facial images as personal data,
skin tone used only for measurement.

### → Literature – Technology Review (1,500 words)
Draw from proposal §2 (Related Work and Positioning), which already frames three
literatures and the gap. Add a **Technology Review** subsection (PyTorch, timm,
transformers, scikit-learn, Streamlit, Kaggle GPU) — the template requires it
even though the proposal does not have one.

### → Conclusion (1,500 words)
Includes required **Future Work** and **Reflection** subsections.

- *Future work:* obtain labelled Fitzpatrick17k to complete Arm A and RQ6;
  stricter grade-specific relevance metric; per-architecture LR sweep; more
  seeds for the GroupDRO stability lead.
- *Reflection:* the ITA failure and its correction is the strongest material
  here — a measurement error caught by visual validation, which changed the
  shape of the project. Examiners reward this.

## 3. Style requirements — check before submitting

The template is explicit:

- **Third-person observational voice only.** No "I", "we", "my", "our", "us".
  The existing drafts in `reports/` are mostly passive/third-person already but
  **must be checked** — search for first-person pronouns before submitting.
- 12pt Arial or Calibri, **1.5 line spacing**, justified, ≥2 cm margins
- Every figure and table **captioned and numbered** (feeds the automated Lists)
- **IEEE reference style** — currently no formal reference list exists; this is
  a genuine gap that needs building
- Each section starts on a new page
- All text black

**Word count excludes** figures, tables, captions, references, and appendices —
so pushing detail into tables is both allowed and encouraged.

## 4. The build pipeline already exists

`D:\Final year project\Report\` contains:

- **`build_report.py`** — injects content into this exact template, strips the
  blue guidance notes and "Insert your text here" placeholders, inserts inline
  figures with numbered captions that feed the automated List of Figures, and
  preserves the template's styles
- **`report_content.py`** — the content held as Python data structures
  (`ACKNOWLEDGEMENTS`, `ABSTRACT`, `REFERENCES`, `APPENDICES`, …)

**This is directly reusable.** The approach: write a new `report_content.py`
for this project, point `FIGDIR` at `D:\sowmiya_FYP\reports\figures`, change
`OUT` to the new filename, and run it. That removes all the manual Word
formatting work — captions, numbering, styles and the figure list are handled.

## 5. Figures available (22, all in `reports/figures/`)

Strongest candidates for inclusion, in priority order:

| Figure | Use in |
|---|---|
| `ita_montage_method_A.png` | Implementation / Results — the pivotal evidence |
| `ita_montage_method_B.png` | Implementation — the corrected estimator |
| `ita_method_distributions.png` | Implementation — the three distributions |
| `rq4_tone_sensitivity.png` | Results — RQ4, the causal finding |
| `rq2_cost_of_error.png` | Results — RQ2 headline |
| `rq5_representation.png` | Results — the 27:1 finding |
| `rq5_pih_coverage.png` | Results — PIH concern coverage |
| `arch_comparison.png` | Results — architecture comparison |
| `rq3_tradeoff_curve.png` | Results — mitigation null |
| `armB_fair_ece.png` | Results — RQ1 calibration gap |
| `abstention_curve.png` | Implementation — threshold derivation |
| `sentiment_by_tone.png` | Results — sentiment fairness |

That is 12 figures, which is a sensible number for a 10,000-word report.

## 6. What is genuinely missing and must be written

1. **Abstract** (500) — write last
2. **Introduction** (1,000) — adapt from proposal §1, §3, §11
3. **Literature Review** (1,000 of the 1,500) — adapt from proposal §2
4. **Technology Review** (500) — new, straightforward
5. **Related Works** subsection in Evaluation — new, required by template
6. **Conclusion + Future Work + Reflection** (1,500) — new
7. **IEEE reference list** — new, and non-trivial; the proposal cites work
   descriptively but there is no formal bibliography yet
8. **Appendices A–D** — proposal, project management evidence, artefact/dataset
   link, screencast link

**Realistic assessment:** items 1–6 are roughly 4,500 words of new writing, plus
the reference list. The results material is already drafted and needs
reorganising rather than writing. That is achievable in two days, but only if
the reference list is started early — it is the task most likely to be
underestimated.
