# Draft email to supervisor

**Subject:** Dissertation progress — five of six research questions complete, one data blocker

---

Dear [Supervisor],

I wanted to update you on the dissertation and flag one problem I'd value your
advice on.

**Where things stand.** The pipeline is built end to end and five of the six
research questions are answered. Both corpora are cleaned and profiled, the
acne severity classifier is trained, the recommender and the
severity-to-ingredient mapping table are complete, and the conversational
demonstrator runs.

**Headline results so far**

- **Severity classifier (Arm B).** Fine-tuned ResNet-50 on ACNE04, repeated
  5-fold cross-validation: **accuracy 0.766, macro-F1 0.746**. Weakest on the
  severe grades, which are also the rarest.

- **RQ1 — fairness audit.** No statistically resolved accuracy gap between the
  Light and Medium tone bands (−0.020, 95% CI [−0.070, +0.026]). There is a
  calibration gap (ECE 0.120 vs 0.092): the model is more confidently wrong on
  the smaller group.

- **RQ4 — counterfactual tone probe.** This is the most interesting result.
  Holding lesion content fixed and shifting only skin tone in CIELAB,
  predictions do move: **6.8% of images change severity grade** under a −20°
  ITA shift, with a clean dose–response (3.6% at −10°) and a zero-shift control
  at exactly 0.0%. So the model *is* causally sensitive to tone even though the
  aggregate audit shows no disparity. I think that contrast is the strongest
  methodological point in the project: **a model can be tone-sensitive at the
  mechanism level while appearing tone-blind in aggregate metrics.**

- **RQ2 — propagation.** Running each subject through the recommender twice,
  once on the true grade and once on the predicted grade, a misclassified user
  receives a substantially different product list (Jaccard overlap 0.515
  against 1.000 for correctly classified users). Propagation is therefore real
  and quantified — but it is **not** tone-differential on this corpus; the
  Light–Medium gaps are either unresolved or too small to change what a user is
  shown.

- **RQ3 — mitigation.** Six strategies × three seeds. No strategy demonstrably
  improves on the unmitigated baseline. The instructive part is that
  seed-to-seed variation (macro-F1 0.702–0.765 under identical settings)
  exceeds every between-strategy difference, so single-seed mitigation results
  would be noise.

- **RQ5 — review corpus audit.** Reviewer representation in the Sephora corpus
  is roughly **27:1 light-to-deep skin tone**. Rating and price differences by
  tone are statistically significant but negligible in effect size. Reviewers
  with deeper tones raise hyperpigmentation concerns at 6.7% against 2.9% for
  lighter tones, while contributing under 3% of reviews.

**An unplanned finding that changed the shape of the project.** My first ITA
implementation classified ACNE04 as ~59% dark-skinned, which is implausible for
a predominantly Chinese corpus. I validated it by rendering montages of faces
grouped by estimated tone, and it was clearly wrong — the "dark" images were
light-skinned faces in shadow or reddened by inflammation. After correcting the
estimator the median ITA is 37° (Fitzpatrick III), which matches the
demographic. Critically, **under every estimator ACNE04 contains no genuinely
dark-skinned subjects.** That triggered the go/no-go pivot in section 10.1 of
the proposal: the tone range is compressed, and I now report that as a finding
in its own right rather than as an obstacle.

**The blocker, and my question for you.** Because of that, Arm A
(Fitzpatrick17k, with ground-truth Fitzpatrick labels) has become the primary
powered fairness audit rather than a supporting one — and I cannot obtain a
usable copy. Every public mirror I have found is either images without the
label file, or labels without the images, and the BTTAI competition version is
invitation-only. The training code is written and waiting.

**Would the department have access to Fitzpatrick17k with its metadata CSV, or
could you advise on the right route to request it?** That single dataset
unblocks both Arm A and RQ6 (the cross-arm comparison). Everything else is
complete.

**Next steps.** Writing up the results chapter around the six research
questions, and running the architecture comparison (EfficientNet, ViT) against
the ResNet-50 baseline.

I'd be glad to walk through any of this in more detail — particularly the RQ4
result, where I'd value your view on how far the causal claim can be pushed
given the probe manipulates rendered tone rather than real subjects.

Best wishes,

[Your name]
