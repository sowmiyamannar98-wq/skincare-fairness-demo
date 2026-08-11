# MSc Data Science — Dissertation Proposal

## Does Bias Propagate? Fairness in Automated Acne Severity Assessment and Downstream Skincare Recommendation

**Student:** [Your name] | **Student number:** [Number]
**Supervisor:** [Professor name]
**Date:** [Date]

---

## 1. Motivation

AI-driven skin analysis has moved rapidly from research into consumer products. Major beauty retailers now offer "skin diagnostic" tools that photograph a user's face, infer skin concerns, and recommend products on that basis. These systems are commercially significant and increasingly trusted by consumers, yet they are trained predominantly on datasets with limited representation of darker skin tones.

The consequences of that imbalance are not evenly distributed. A user whose concern is misclassified does not simply receive a wrong label — they are shown a different set of products, potentially fewer options, potentially more expensive ones, potentially products unsuited to their actual needs. The harm is realised not at the point of classification but at the point of recommendation.

Two mature literatures bear on this, and they do not currently meet. Fairness auditing of dermatological image models is well established, consistently reporting accuracy gaps across skin tone groups in lesion and melanoma classification. Separately, methods for analysing fairness across multi-stage machine learning pipelines have developed rapidly, including system-level frameworks for compositional recommenders, fairness guarantees for two-stage retrieval-and-ranking architectures, and evidence that bias introduced upstream propagates downstream and cannot reliably be corrected by rebalancing at the later stage.

The gap lies at their intersection. Dermatological fairness work is almost entirely diagnostic, concerned with disease classification rather than cosmetic assessment. Pipeline fairness work is largely situated in domains such as employment, credit, and content ranking. Deep learning systems for skincare product recommendation from facial images do exist — Lee et al. (2024) combine facial skin condition analysis with cosmetic ingredient analysis — but they are evaluated on accuracy and treat bias only as acknowledged future work, anticipating that broader datasets will mitigate it rather than measuring whether it is present.

This project occupies that intersection. It adopts established pipeline-fairness methodology and applies it, for the first time to my knowledge, to consumer skincare recommendation: constructing a two-stage system — image-based severity assessment feeding a review-driven product recommender — and measuring fairness at both stages and across the boundary between them. The contribution is therefore not a new fairness method but the transfer of an existing methodological apparatus into a commercially widespread application domain where it has not been applied, and where the affected users are consumers rather than patients. The multi-modal design is not decorative: the research question cannot be answered without both halves.

---

## 2. Related Work and Positioning

The project sits at the meeting point of three bodies of work, and is positioned explicitly against each.

**Fairness in dermatological AI.** A substantial literature audits skin lesion and melanoma classifiers across Fitzpatrick tone groups, consistently finding degraded performance on darker skin, driven by training corpora dominated by lighter tones. Recent work extends this to the granularity of tone labelling itself and to generative augmentation as mitigation. This literature supplies the methodological grounding for the audit in RQ1 — but it is uniformly diagnostic. It addresses clinical harm to patients, not allocative harm to consumers, and it stops at the classifier.

**Fairness across multi-stage pipelines.** A parallel literature examines how bias behaves when models are composed. Relevant strands include system-level fairness frameworks for compositional recommender architectures that track propagation between retrieval and ranking; distribution-free guarantees for two-stage recommender pipelines in which a fast, less accurate first stage constrains what the second stage can surface; studies of cumulative effects when mitigation is applied at multiple pipeline stages; and evidence from deep metric learning that bias in upstream representations persists into downstream tasks even when downstream data are rebalanced. This supplies the analytical framework for RQ2. Its application domains, however, are predominantly employment, credit, and content ranking.

**Automated skincare recommendation.** Systems combining facial skin analysis with product recommendation have been built and published, notably Lee et al. (2024), who pair facial condition analysis with cosmetic ingredient analysis. Such work is evaluated on recommendation accuracy. Bias is raised as a limitation to be addressed by future data collection rather than as a quantity to be measured. Lee et al. additionally report severe imbalance in product-level concern labelling — acne-labelled products numbering in the low hundreds against several thousand for wrinkles — which motivates the corpus audit in RQ5.

**The gap.** Pipeline-fairness methodology has not been applied to consumer skincare recommendation, and existing skincare recommenders have not been audited for tone-related disparity at all, let alone for propagation. This project's contribution is the transfer of an established methodological apparatus into a domain where it is absent, on a system class already deployed commercially at scale. This is a contribution of application and evidence rather than of novel method, and is claimed as such.

*Note: this positioning is based on a preliminary search and will be formalised through a systematic review in Phase 3. The novelty claim is deliberately scoped to the intersection rather than to propagation analysis in general, which is well established.*

---

## 3. Aim and Research Questions

### Aim

To determine, with quantified uncertainty, whether disparities in automated skin assessment across skin tone groups propagate into the product recommendations that depend on them; and to test whether fairness findings established on a large, tone-labelled corpus transfer to the smaller, tone-estimated facial data on which such consumer systems are actually built.

### Two-arm design

The fairness investigation is conducted across two complementary arms, because no single public dataset supports both statistical power and application realism.

| | Corpus | Contributes | Constraint it removes | Integrated into the application? |
|---|---|---|---|---|
| **Arm A** — reference audit | Fitzpatrick17k (16,577 images) | Well-powered audit on ground-truth Fitzpatrick labels; full mitigation comparison | Statistical power; tone-label validity | No — offline experiment only |
| **Arm B** — applied audit | ACNE04 (1,457 images) | Audit of the deployed facial model on ITA-derived tone groups | Domain realism; matches consumer input | Yes — this is the model the demonstrator runs |

Arm A has the power and the labels but the wrong image domain; Arm B has the right domain but limited sample size and estimated labels. Each compensates for the other's principal weakness, and the comparison between them constitutes RQ6. **Arm A is deliberately confined to the fairness analysis: it is not connected to the recommender, the ingredient mapping, or the conversational interface.** This boundary is stated explicitly as a scope control.

### Scope of the visual task

The vision model performs **acne severity assessment**, which is what the available annotated facial data support. Additional attributes that materially affect product suitability — skin type (oily, dry, combination), sensitivities, and budget — are **collected conversationally from the user**, because no public facial dataset labels them and inferring them from pixels would not be defensible. This division is a deliberate methodological position, stated explicitly rather than obscured, and it mirrors how deployed systems operate. The conversational interface described in Section 7.7 is the mechanism by which these attributes are elicited.

### Research Questions

**RQ1.** Does automated acne severity classification from facial images perform equitably across skin tone groups, measured by disaggregated error rates and calibration, with disparities reported as effect sizes with confidence intervals?

**RQ2.** Do classification disparities propagate into downstream recommendation outcomes — specifically the relevance, diversity, catalogue coverage, and price distribution of recommended products?

**RQ3.** Which bias mitigation strategies most effectively reduce disparity, and what is the cost to overall predictive performance?

**RQ4.** Is the model's output causally sensitive to skin tone itself, independent of the underlying condition?

**RQ5.** Is the product review corpus itself representationally biased — in review volume, sentiment, rating, or price — with respect to reviewer-reported skin tone and the concerns associated with it?

**RQ6.** Do fairness findings established on a large corpus with ground-truth tone labels (Arm A) replicate on a small facial corpus with estimated tone labels (Arm B) — and if not, what does that imply for the external validity of single-corpus fairness audits?

---

## 4. Objectives

1. Acquire, clean, and profile all three corpora through a substantive exploratory data analysis, quantifying class balance, tone distribution, missingness, review volume, and price structure.
2. Derive objective skin tone groupings for facial images using Individual Typology Angle (ITA), and characterise the resulting distribution.
3. **Assess dataset viability at an early go/no-go gate** (Section 10.1) before committing to the primary analysis path.
4. Train and compare deep learning classifiers on both arms, establishing strong baselines before optimisation.
5. Conduct the Arm A reference audit at full depth: disaggregated metrics, calibration, counterfactual probe, and mitigation comparison, with bootstrapped confidence intervals throughout.
6. Conduct the Arm B applied audit on the deployed facial model, and compare findings across arms to address RQ6.
7. Construct a documented, literature-referenced mapping from severity grade and declared attributes to skincare ingredient classes.
8. Build a content-based recommender over the product review corpus using that mapping.
9. Measure fairness at the recommendation stage and quantify propagation from stage one to stage two.
10. Audit the review corpus independently for representational bias (RQ5).
11. Deliver a conversational demonstrator application evidencing the pipeline end to end, with recommendation logic held strictly outside the language model.

---

## 5. Datasets

Three corpora are used. Two are integrated into the application pipeline; the third supports the reference fairness arm and is deliberately kept separate from it.

### 5.1 ACNE04 — Facial Image Analysis

- **Source:** Wu et al., *Joint Acne Image Grading and Counting via Label Distribution Learning*, ICCV 2019.
- **Contents:** Facial photographs annotated by dermatologists for acne severity under the Hayashi criterion, with bounding-box lesion annotations.
- **Role:** Training and evaluation of the visual severity model; this is the model deployed in the demonstrator.
- **Rationale:** ACNE04 consists of facial images of the kind a consumer application would actually receive. Clinical dermatology corpora, though richer in tone labels, contain lesion photographs of body sites under clinical conditions; a model trained on them would return diagnostic categories when shown a selfie, which is technically invalid and ethically inappropriate for a cosmetic recommendation tool.
- **Scale:** 1,457 images with 18,983 annotated lesions. This is small, and it is the principal motivation for the two-arm design: stratified simultaneously by severity class and tone group, per-cell counts become too thin to support well-powered interval estimates. Arm B is therefore treated as an applied replication rather than the primary audit.
- **Known limitations:** the image population is predominantly Chinese, compressing available tone range; severity classes are imbalanced, with severe grades under-represented. Both are quantified in EDA and drive the decision gate in Section 10.1. Expert re-annotated labels (Acne04-v2) and published synthetic augmentation for under-represented severity grades are available and will be evaluated as remedies for the class imbalance.

### 5.2 Sephora Products and Skincare Reviews — Text and Recommendation

- **Source:** Public Kaggle dataset (Nadyinky).
- **Contents:** Approximately 8,000 skincare products and 1 million customer reviews, with ingredient lists, price, brand, category, star ratings, review text, and reviewer-reported attributes including skin type and skin tone.
- **Role:** Sentiment and aspect mining; ingredient parsing; the product catalogue over which recommendations are ranked; and, via reviewer-reported attributes, the corpus for the independent representational audit in RQ5.
- **Rationale:** Its scale supports meaningful NLP; its ingredient and category metadata support content-based recommendation without a dense user–item interaction matrix; and its reviewer-reported skin type provides a natural bridge from assessed need to suitable product.
- **Known limitation:** the catalogue is prestige-only, containing no mass-market or budget products. The recommender's scope is therefore premium skincare, stated as a boundary condition — and one that additionally motivates the price-disparity analysis in RQ2.

### 5.3 Fitzpatrick17k — Reference Fairness Arm

- **Source:** Groh et al., publicly released clinical dermatology corpus.
- **Contents:** 16,577 images annotated with Fitzpatrick skin tone types I–VI across a long tail of skin conditions.
- **Role:** Arm A only — the well-powered reference audit. Provides ground-truth tone labels and sufficient sample size for the full mitigation comparison and calibration analysis that ACNE04 cannot support.
- **Rationale:** it supplies precisely what Arm B lacks. Ten times the images, real tone labels rather than estimates, and enough per-group density for confidence intervals narrow enough to be informative.
- **Explicit scope boundary:** Fitzpatrick17k is **not** used to train any model deployed in the demonstrator, and is not connected to the recommender. Clinical lesion photography is the wrong input distribution for a consumer selfie application, and conflating the two would be both technically invalid and ethically inappropriate.
- **Known limitations:** documented label noise, variable image quality, and incomplete image availability at fetch time. Conditions are collapsed to a coarser grouping rather than modelling the full long tail. The retrieval success rate is recorded and reported.

### 5.4 Skin Tone Derivation

Fitzpatrick17k supplies tone labels directly; ACNE04 does not, and no public facial corpus does. Skin tone groups are derived computationally using **Individual Typology Angle (ITA)**, calculated from non-lesional skin pixels in CIELAB colour space following established practice in dermatological fairness literature. ITA values are binned into tone categories and validated by manual inspection of a stratified sample, with inter-rater agreement reported on that sample. ITA is an objective proxy rather than ground truth, and this is treated as a methodological limitation throughout, including sensitivity analysis under alternative bin boundaries.

---

## 6. Exploratory Data Analysis

A dedicated analysis phase precedes modelling and forms a substantive chapter of the dissertation. It is not preparatory housekeeping; several of its outputs are findings in their own right.

**Image corpus.** Severity class distribution; ITA distribution overall and cross-tabulated with severity; sample counts per tone group with confidence intervals on group proportions; image quality, resolution, and lighting variation, since illumination confounds ITA estimation and is examined explicitly.

**Review corpus.** Review volume per product and its skew; rating distributions and rating–sentiment agreement; missingness across reviewer-reported fields; price distribution by category and brand; ingredient list length, format inconsistency, and vocabulary size; representation of reviewer skin tone and skin type, which supplies the baseline for RQ5.

**Statistical framing.** Distributional comparisons are reported with appropriate tests and effect sizes rather than raw differences alone, and group proportions are reported with intervals so that later fairness comparisons rest on a documented account of sampling uncertainty.

---

## 7. Methodology and Models

### 7.1 Image Modules — Both Arms

| Stage | Approach |
|---|---|
| Baseline | ResNet-50, ImageNet-pretrained, fine-tuned |
| Comparison | EfficientNet-B0/B3 and a Vision Transformer (ViT-Base) |
| Preprocessing | Face detection and cropping, standardised resizing, colour-consistent augmentation |
| Framework | PyTorch with `timm` and Hugging Face `transformers` |

The same architectural family and training protocol are applied to both arms, so that cross-arm differences in RQ6 reflect the corpora rather than incidental methodological divergence. Arm A classifies skin condition on Fitzpatrick17k; Arm B classifies acne severity on ACNE04. Data are split with stratification by both outcome class and tone group, so that per-group evaluation is possible at test time. Augmentation is applied cautiously: colour and brightness transformations must not distort the tone signal on which the fairness analysis depends. Model comparison uses repeated runs across multiple random seeds, with performance reported as a mean and interval rather than a single figure, so that claimed differences between architectures are distinguishable from run-to-run variance. Given ACNE04's limited size, Arm B uses repeated stratified cross-validation rather than a single held-out split, so that disparity estimates draw on the full corpus rather than a 292-image test set.

### 7.2 Fairness Analysis

- **Disaggregated performance:** accuracy, precision, recall, and macro-F1 per tone group, each reported with **bootstrapped confidence intervals**. This matters acutely here: fairness metrics computed on small groups are unstable, and a point estimate alone can imply a disparity that resampling does not support.
- **Group fairness metrics:** equal opportunity difference, equalised odds difference, and demographic parity difference, likewise bootstrapped.
- **Effect sizes and significance:** disparities reported as effect sizes with intervals, with tests chosen appropriately for paired model comparisons and correction applied for multiple comparisons across groups and metrics.
- **Calibration:** per-group Expected Calibration Error and reliability diagrams. Under-represented groups frequently exhibit worse calibration — confidently incorrect predictions — which accuracy alone does not reveal, and which directly justifies the abstention threshold in the deployed system.
- **Counterfactual tone probe (RQ4):** controlled tone shifts applied to test images in CIELAB space, holding lesion content constant, to test whether predictions change with tone alone. This distinguishes genuine tone sensitivity from confounding with condition prevalence, and provides causal evidence that correlational auditing cannot.
- **Pre-committed protocol:** the metric set, group definitions, and testing procedure are fixed and documented **before results are inspected**, so that reporting cannot drift toward whichever measure proves most flattering.
- **Mitigation comparison:** class and group reweighting, oversampling of under-represented groups, loss-based approaches (focal loss, group-distributionally robust optimisation), and targeted augmentation, reported on a shared accuracy–fairness trade-off curve with uncertainty bands.

### 7.3 Text Module — Review Mining

| Task | Approach |
|---|---|
| Sentiment classification | DistilBERT or RoBERTa fine-tuned on review text |
| Aspect extraction | Rule-based and embedding-based extraction of concern-relevant mentions (breakouts, irritation, hydration) |
| Ingredient parsing | Rule-based normalisation of free-text ingredient lists into a controlled vocabulary |
| Representation | TF-IDF and sentence embeddings for product similarity |

Ingredient parsing is deliberately rule-based rather than learned: the data are inconsistent free text, and a documented deterministic pipeline is both more tractable and more defensible than an opaque model.

### 7.4 Severity-and-Attribute to Ingredient Mapping

A documented mapping table links assessed severity and user-declared attributes to ingredient classes, referenced to dermatological literature rather than marketing sources — for example, mild acne to salicylic acid and niacinamide, moderate to benzoyl peroxide or adapalene-adjacent formulations, with declared dryness or sensitivity constraining toward ceramides, hyaluronic acid, and glycerin and away from high-irritancy actives. This table is a deliberate design artefact and is presented and defended as such, since it constitutes the interpretable seam between the two modules.

### 7.5 Recommendation Module

A content-based ranker scores candidate products on ingredient match, aggregate review sentiment, star rating, and compatibility with declared skin type and budget. A learned collaborative-filtering approach is explicitly rejected: the data do not support it, it would consume disproportionate time, and an opaque ranker would obstruct the fairness analysis that is the project's purpose.

### 7.6a Cross-Arm Comparison (RQ6)

Findings are compared across arms along three dimensions: whether the *direction* of disparity agrees; whether the *ordering* of mitigation strategies by effectiveness is preserved; and whether disparities detectable at Arm A's sample size remain detectable at Arm B's. The third is treated carefully — a non-significant result in Arm B is reported as a power limitation, evidenced by interval width, rather than as contradiction of Arm A. Where ITA-derived groups can be compared against Fitzpatrick labels on Arm A, agreement between the two is quantified, providing a direct validation of the tone-estimation procedure used throughout Arm B.

### 7.6 Review Corpus Audit (RQ5)

Independently of the image pipeline, the review corpus is analysed for representational bias using reviewer-reported skin tone: review volume by tone; rating and sentiment distributions by tone; price of highly rated products by tone; and coverage of concerns disproportionately affecting darker skin, notably post-inflammatory hyperpigmentation. This arm is deliberately independent of the image model, so that a substantive fairness contribution survives even if the image tone range proves too narrow for a well-powered audit.

---

### 7.7 Conversational Interface Architecture

The system is delivered as a conversational application rather than a static upload form. This is a methodological choice, not a presentational one: the pipeline requires attributes the image cannot supply — skin type, sensitivities, existing active ingredient use, and budget — and dialogue is the natural mechanism for eliciting them. The image model assesses what is visible; the conversation supplies what is not.

**Architectural constraint.** Introducing a language model into a fairness-critical pipeline adds a component with its own documented demographic biases, situated between the audited system and the user. If that component influenced recommendations, the propagation analysis in RQ2 would no longer characterise the system it claims to measure. A strict separation of responsibilities is therefore imposed:

| Component | Responsibility | Determines recommendations? |
|---|---|---|
| Language model | Question phrasing; parsing free-text replies into structured slots; verbalising the final output | No |
| CNN classifier | Acne severity from the uploaded image | Yes |
| Mapping table | Severity and declared attributes to ingredient classes | Yes |
| Ranking function | Product scoring and ordering | Yes |

The language model receives an already-ranked list and explains it. It does not select, reorder, filter, or introduce products, and it may not make efficacy claims beyond those supported by the mapping table and retrieved product data. Every recommendation is therefore traceable to a deterministic, inspectable path — which is what makes the pipeline auditable at all. A rules-based dialogue manager is retained as a fallback, and the system remains fully functional without the language model, which sacrifices fluency but nothing evaluable.

**Safety behaviours.** A conversational interface invites requests a form does not, principally medical ones: questions about lesion malignancy, prescription treatments, or diagnosis. Defined refusal-and-redirect behaviour is specified for these cases and tested against a purpose-built set of adversarial prompts, with transcripts reported in an appendix as evidence of boundary testing. The abstention threshold from the calibration analysis applies here too: below it, the system states that it cannot assess the image with confidence rather than producing a recommendation.



### 8.1 Research Outcomes

1. **A quantified fairness audit** of facial acne severity classification across ITA-derived tone groups, reporting calibration alongside accuracy and expressing every disparity with uncertainty.
2. **A measurement of bias propagation** from classification into recommendation outcomes — the principal contribution, applying established pipeline-fairness methodology to a domain where, per Section 2, it has not previously been applied.
3. **A comparative evaluation of mitigation strategies** as a trade-off curve rather than a single-point claim.
4. **Causal evidence on tone sensitivity** via the counterfactual probe.
5. **An independent representational audit** of a large commercial review corpus (RQ5).
6. **A cross-arm replication result** (RQ6) speaking to the external validity of single-corpus fairness audits — including a direct quantification of agreement between ITA-estimated and ground-truth Fitzpatrick tone labels.
7. **A characterisation of dataset limitations** in publicly available facial skin corpora, quantifying the tone-range and sample-size constraints that bound work of this kind.

### 8.2 Artefacts

- A reproducible codebase covering data pipeline, EDA, model training, fairness evaluation, and recommendation.
- A documented, literature-referenced mapping table.
- A conversational demonstrator application (Streamlit or Gradio): image upload, severity assessment, dialogue-based attribute elicitation, and explained product recommendations, architected per Section 7.7 so that all recommendation logic remains deterministic and auditable.
- An adversarial prompt set and resulting transcripts, evidencing tested refusal behaviour on medical requests.
- The dissertation, with a results chapter structured around the five research questions.

### 8.3 What Success Looks Like

Success is defined by the quality and honesty of the evidence, not the size of the disparity found. A rigorous audit reporting **no significant disparity**, with intervals demonstrating the study was adequately powered to detect one, is a valid and reportable outcome. The pre-committed protocol exists precisely so that a null result cannot be quietly discarded in favour of a more flattering framing.

---

## 9. Evaluation Plan

| Component | Method |
|---|---|
| Classification | Accuracy, macro-F1, per-group metrics, per-group ECE, confusion analysis; multi-seed runs with intervals |
| Fairness | Equal opportunity, equalised odds, demographic parity; bootstrapped CIs, effect sizes, multiple-comparison correction |
| Sentiment | Standard classification metrics on a held-out labelled subset |
| Recommendation relevance | Precision@k against a literature-validated reference set of acceptable ingredient classes |
| Recommendation fairness | Catalogue coverage, list diversity, and price distribution, disaggregated by tone group with intervals |
| Baseline comparison | Contrast against a popularity-only recommender, to evidence that the pipeline adds value over trivial ranking |
| Pipeline | Propagation analysis: recommendation quality conditioned on correct versus incorrect upstream prediction |
| Cross-arm (RQ6) | Agreement in disparity direction and mitigation ranking across arms; ITA-versus-Fitzpatrick label agreement; power assessment via interval width |
| Review corpus | Representation, sentiment, rating, and price distributions by reviewer-reported tone |
| Conversational interface | Slot-filling accuracy against annotated test dialogues; refusal rate on an adversarial medical-request set; verification that verbalised output contains no products absent from the ranked list |

There is no ground truth for "the correct product." This is addressed by evaluating against a defensible reference standard of acceptable ingredient classes rather than exact product matches, and by reporting secondary metrics that do not depend on a single correct answer.

---

## 10. Risks and Mitigation

### 10.1 Primary risk and early decision gate

The tone range available in ACNE04 is load-bearing for RQ1 and RQ2. This is resolved at an explicit **go/no-go gate in the first phase of work**, before modelling begins: ITA is computed across the corpus, the distribution is plotted, and per-group sample counts are established with confidence intervals on group proportions.

- **If the distribution supports a well-powered audit** — the primary path proceeds as specified.
- **If the range is compressed** — the project pivots immediately, without loss of contribution, by (a) adding a tone-labelled dermatological corpus as a separate offline fairness arm where tone labels are ground truth rather than estimated, and (b) shifting analytical weight toward the review corpus audit (RQ5). The compressed range is then itself reported as a quantified finding about the state of publicly available facial datasets.

Committing to this decision point in advance is what prevents the risk from materialising late, when no pivot is affordable.

### 10.2 Further risks

| Risk | Mitigation |
|---|---|
| Severity class imbalance in ACNE04 | Quantified in EDA; addressed through class weighting and resampling, with macro-averaged metrics reported |
| ITA estimation confounded by image illumination | Illumination variation examined in EDA; sensitivity analysis under alternative bin boundaries; manual validation on a stratified sample |
| Scope overruns across modules | Baseline-first strategy: a thin end-to-end slice before any module is deepened; off-the-shelf sentiment models if time is short |
| Ingredient parsing more laborious than expected | Restricted to a prioritised subset of active ingredients relevant to the mapping table |
| No significant disparity found | Pre-committed metrics and protocol; reported as a legitimate finding with power addressed via interval width |
| Two arms executed shallowly rather than one properly | Arm A is the primary audit and is completed in full before Arm B begins; Arm B is scoped as replication, not a second full study |
| Third corpus expands scope by attachment to the pipeline | Hard boundary in Section 5.3: Fitzpatrick17k touches the fairness analysis only, never the recommender, mapping table, or application |
| Demonstrator displaces research work | Application built last and strictly timeboxed; ugly-but-functional accepted |
| Language model introduces unaudited bias into the pipeline | Strict separation of responsibilities (7.7): the model phrases and parses but never determines recommendations; rules-based dialogue retained as a functional fallback |
| Users direct medical questions at the conversational interface | Specified refusal-and-redirect behaviour, tested against an adversarial prompt set and reported in an appendix |

---

## 11. Ethics, Safety, and Scope

- The system is a **cosmetic recommendation tool, not a medical device**. It does not diagnose. This is stated prominently in the interface and throughout the dissertation.
- The classifier **abstains below a confidence threshold** rather than producing a low-confidence guess, with the threshold justified by the per-group calibration analysis rather than chosen arbitrarily. The conversational interface surfaces this abstention explicitly rather than substituting a plausible-sounding answer.
- The conversational interface **declines medical requests** — diagnosis, malignancy assessment, prescription guidance — and redirects to qualified care. This behaviour is specified in advance and tested, not left to the language model's discretion.
- All data are pre-existing public research datasets; no new facial images are collected. Departmental ethics guidance will be followed and approval sought where required.
- Facial images are personal data and are handled accordingly: secure storage, no redistribution, and no retention of images submitted to the demonstrator.
- Skin tone is treated as a sensitive attribute used solely for fairness measurement, never as a recommendation input.

---

## 12. Indicative Plan

| Phase | Focus |
|---|---|
| 1 | Data acquisition and cleaning; ITA computation; **go/no-go gate (10.1)** |
| 2 | Exploratory data analysis on both corpora; split design |
| 3 | Literature review; mapping table construction |
| 4 | Arm A baseline classifier; thin end-to-end pipeline |
| 5 | Arm A reference audit: calibration, counterfactual probe, full mitigation comparison |
| 6 | Arm B applied audit on ACNE04; cross-arm comparison (RQ6) |
| 7 | Review mining, recommender implementation, review corpus audit |
| 8 | Propagation analysis and recommendation fairness evaluation |
| 9 | Conversational demonstrator; adversarial prompt testing (timeboxed to one week) |
| 10 | Writing, revision, submission |

---

## 13. Why This Merits MSc-Level Study

The contribution is not the construction of a working system, which would be an engineering exercise. It is the investigation of a specific and under-examined question: whether fairness measured at one stage of an AI pipeline is preserved, amplified, or obscured at the next. Answering it requires the multi-modal system as an instrument, applies established fairness methodology in a novel configuration, and rests on quantified uncertainty rather than point estimates — so that its findings are informative whether the disparity observed proves large, small, or absent. The two-arm design adds a second-order contribution: by running comparable audits on a large tone-labelled corpus and a small tone-estimated one, it tests whether conclusions of the kind the fairness literature routinely draws from a single dataset survive a change of corpus.
