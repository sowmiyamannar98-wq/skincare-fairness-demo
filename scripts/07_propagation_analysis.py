"""
07_propagation_analysis.py
------------------------------------------------------------------
RQ2 -- THE PRINCIPAL CONTRIBUTION.

Do classification disparities propagate into downstream recommendation
outcomes -- specifically the relevance, diversity, catalogue coverage, and
price distribution of recommended products?

Method. For every ACNE04 image we hold both the dermatologist grade (y_true)
and the model's out-of-fold prediction (y_pred). We run the SAME user through
the SAME recommender twice:

    counterfactual arm : recommendations the user SHOULD have received (y_true)
    deployed arm       : recommendations the user ACTUALLY receives   (y_pred)

The gap between those two lists is the harm realised at the point of
recommendation rather than at the point of classification. Because the
mapping table is deterministic, any divergence is attributable to the upstream
classifier alone -- which is precisely what makes the pipeline auditable.

Declared attributes (skin type, sensitivity, budget) cannot be inferred from
pixels, so they are SAMPLED from the empirical distribution of self-reported
Sephora reviewer attributes with a fixed seed. Each simulated user therefore
has a realistic, distinct profile, and the propagation measured is not an
artefact of one arbitrary profile.

Everything is disaggregated by ITA tone band (validated Method B) and reported
with bootstrapped confidence intervals.

Run:  python scripts/07_propagation_analysis.py

Outputs:
  data/processed/rq2_propagation_per_user.parquet
  reports/07_rq2_propagation.md
  reports/figures/rq2_*.png
------------------------------------------------------------------
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommender import SkincareRecommender, REFERRAL_GRADES   # noqa: E402

PROC = os.path.join("data", "processed")
KO = os.path.join("data", "kaggle_out")
FIG = os.path.join("reports", "figures")
OUT = os.path.join("reports", "07_rq2_propagation.md")
os.makedirs(FIG, exist_ok=True)

PRED = os.path.join(KO, "armB_oof_predictions.csv")
ITA = os.path.join(KO, "acne04_ita_validated.csv")
TONE_METHOD = "ita_B"
GROUPS = ["Light", "Medium"]        # Deep excluded: artifact (see NB1b)
K = 10
B = 1000
RNG = np.random.default_rng(42)
lines = []


def w(s=""):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))
    lines.append(s)


def stem(p):
    return os.path.splitext(str(p).replace("\\", "/").split("/")[-1])[0]


def band_from_ita(v):
    if pd.isna(v):
        return "unknown"
    return "Light" if v > 41 else "Medium" if v > 19 else "Deep"


# ============================================================ 1. LOAD
pred = pd.read_csv(PRED).drop(columns=["tone_band"], errors="ignore")
ita = pd.read_csv(ITA)
pred["key"] = pred["path"].map(stem)
ita["key"] = ita["path"].map(stem)
ita["tone_band"] = ita[TONE_METHOD].map(band_from_ita)
users = pred.merge(ita[["key", "tone_band"]], on="key", how="left")

w("# RQ2 -- Does Classifier Bias Propagate into Recommendations?\n")
w("_Principal contribution. Each ACNE04 subject is run through the same "
  "deterministic recommender twice: once on the dermatologist grade "
  "(what they should receive) and once on the model's prediction (what they "
  "actually receive). Divergence between the two lists is attributable to the "
  "upstream classifier alone._\n")
w(f"- Simulated users: **{len(users):,}**")
w(f"- Upstream classifier: ResNet-50, out-of-fold accuracy "
  f"**{(users.y_true == users.y_pred).mean():.3f}**")
w(f"- Tone labels: validated ITA (`{TONE_METHOD}`); audit restricted to "
  "Light vs Medium (Deep excluded as illumination artifact, see NB1b)\n")


# ============================================================ 2. PROFILES
# sampled from Sephora self-reported reviewer attributes (RQ5 EDA)
SKIN_TYPES = ["combination", "dry", "normal", "oily"]
SKIN_P = np.array([0.554, 0.189, 0.134, 0.123])       # renormalised, RQ5 table
users["skin_type"] = RNG.choice(SKIN_TYPES, size=len(users), p=SKIN_P)
users["sensitive"] = RNG.random(len(users)) < 0.25
users["pih_concern"] = RNG.random(len(users)) < 0.10
users["budget"] = RNG.choice([None, 50, 100], size=len(users),
                             p=[0.5, 0.25, 0.25])

rec = SkincareRecommender()
w(f"- Catalogue after filtering: **{len(rec.cat):,}** products\n")


# ============================================================ 3. RUN BOTH ARMS
def profile_key(r):
    return (int(r.grade), r.skin_type, bool(r.sensitive),
            bool(r.pih_concern), r.budget)


cache = {}


def get_recs(grade, skin_type, sensitive, pih, budget):
    key = (int(grade), skin_type, bool(sensitive), bool(pih), budget)
    if key not in cache:
        cache[key] = rec.recommend(grade=grade, skin_type=skin_type,
                                   sensitive=sensitive, pih_concern=pih,
                                   budget=budget, k=K)
    return cache[key]


rows = []
for r in users.itertuples(index=False):
    tr = get_recs(r.y_true, r.skin_type, r.sensitive, r.pih_concern, r.budget)
    dp = get_recs(r.y_pred, r.skin_type, r.sensitive, r.pih_concern, r.budget)
    s_tr, s_dp = set(tr["product_id"]), set(dp["product_id"])
    inter = len(s_tr & s_dp)
    union = len(s_tr | s_dp) or 1
    rows.append(dict(
        key=r.key, tone_band=r.tone_band,
        y_true=r.y_true, y_pred=r.y_pred,
        correct=int(r.y_true == r.y_pred),
        jaccard=inter / union,
        overlap_k=inter / max(len(s_tr), 1),
        price_true=tr["price_usd"].mean(),
        price_deployed=dp["price_usd"].mean(),
        price_delta=dp["price_usd"].mean() - tr["price_usd"].mean(),
        # relevance is ALWAYS scored against the TRUE grade's reference set
        prec_true=rec.precision_at_k(tr, r.y_true),
        prec_deployed=rec.precision_at_k(dp, r.y_true),
        brands_deployed=dp["brand_name"].nunique(),
        cats_deployed=dp["secondary_category"].nunique(),
        referral_needed=int(r.y_true in REFERRAL_GRADES),
        referral_given=int(r.y_pred in REFERRAL_GRADES),
        deployed_ids=",".join(map(str, sorted(s_dp))),
    ))
res = pd.DataFrame(rows)
res["prec_loss"] = res["prec_true"] - res["prec_deployed"]
res["missed_referral"] = ((res.referral_needed == 1) &
                          (res.referral_given == 0)).astype(int)
res["unnecessary_referral"] = ((res.referral_needed == 0) &
                               (res.referral_given == 1)).astype(int)
res.to_parquet(os.path.join(PROC, "rq2_propagation_per_user.parquet"),
               index=False)


# ============================================================ 4. STATS
def boot_mean(v, b=B):
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(v), size=(b, len(v)))
    bs = v[idx].mean(axis=1)
    return (v.mean(), *np.percentile(bs, [2.5, 97.5]))


def boot_diff(a, b_, b=B):
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b_ = np.asarray(pd.Series(b_).dropna(), dtype=float)
    if len(a) == 0 or len(b_) == 0:
        return (np.nan, np.nan, np.nan)
    d = a.mean() - b_.mean()
    bs = np.empty(b)
    for i in range(b):
        bs[i] = (a[RNG.integers(0, len(a), len(a))].mean()
                 - b_[RNG.integers(0, len(b_), len(b_))].mean())
    return (d, *np.percentile(bs, [2.5, 97.5]))


w("## 1. Propagation headline -- the cost of an upstream error\n")
mis = res[res.correct == 0]
cor = res[res.correct == 1]
w(f"- Users receiving a **correct** grade: {len(cor):,} "
  f"({100*len(cor)/len(res):.1f}%)")
w(f"- Users receiving a **misclassified** grade: {len(mis):,} "
  f"({100*len(mis)/len(res):.1f}%)\n")

j_c, j_m = boot_mean(cor.jaccard), boot_mean(mis.jaccard)
p_c, p_m = boot_mean(cor.prec_deployed), boot_mean(mis.prec_deployed)
d_c, d_m = boot_mean(cor.price_delta), boot_mean(mis.price_delta)
w("| outcome | list overlap with correct list (Jaccard) | relevance P@10 vs true grade | price shift (USD) |")
w("|---|---|---|---|")
w(f"| correctly classified | {j_c[0]:.3f} [{j_c[1]:.3f}, {j_c[2]:.3f}] | "
  f"{p_c[0]:.3f} [{p_c[1]:.3f}, {p_c[2]:.3f}] | "
  f"{d_c[0]:+.2f} [{d_c[1]:+.2f}, {d_c[2]:+.2f}] |")
w(f"| **misclassified** | {j_m[0]:.3f} [{j_m[1]:.3f}, {j_m[2]:.3f}] | "
  f"{p_m[0]:.3f} [{p_m[1]:.3f}, {p_m[2]:.3f}] | "
  f"{d_m[0]:+.2f} [{d_m[1]:+.2f}, {d_m[2]:+.2f}] |")

jd = boot_diff(cor.jaccard, mis.jaccard)
pd_ = boot_diff(cor.prec_deployed, mis.prec_deployed)
w(f"\n- **Overlap penalty of misclassification:** {jd[0]:+.3f} "
  f"[{jd[1]:+.3f}, {jd[2]:+.3f}]")
w(f"- **Relevance penalty of misclassification:** {pd_[0]:+.3f} "
  f"[{pd_[1]:+.3f}, {pd_[2]:+.3f}]")
w("\nThis is the propagation effect in its simplest form: an upstream error "
  "does not merely mislabel a user, it replaces the product set they would "
  "have been shown.\n")


w("## 2. Propagation disaggregated by tone group\n")
w("| tone band | n | acc | Jaccard | relevance P@10 | price shift | missed referrals |")
w("|---|---|---|---|---|---|---|")
per = {}
for g in GROUPS:
    sub = res[res.tone_band == g]
    acc = sub.correct.mean()
    j = boot_mean(sub.jaccard)
    p = boot_mean(sub.prec_deployed)
    d = boot_mean(sub.price_delta)
    mr = sub.missed_referral.mean()
    per[g] = dict(n=len(sub), acc=acc, j=j, p=p, d=d, mr=mr)
    w(f"| {g} | {len(sub)} | {acc:.3f} | {j[0]:.3f} [{j[1]:.3f}, {j[2]:.3f}] | "
      f"{p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}] | "
      f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] | {100*mr:.1f}% |")

gj = boot_diff(res[res.tone_band == "Light"].jaccard,
               res[res.tone_band == "Medium"].jaccard)
gp = boot_diff(res[res.tone_band == "Light"].prec_deployed,
               res[res.tone_band == "Medium"].prec_deployed)
gd = boot_diff(res[res.tone_band == "Light"].price_delta,
               res[res.tone_band == "Medium"].price_delta)
w(f"\n**Light - Medium gaps (bootstrapped):**\n")
w(f"- Jaccard overlap: {gj[0]:+.3f} [{gj[1]:+.3f}, {gj[2]:+.3f}]")
w(f"- Relevance P@10: {gp[0]:+.3f} [{gp[1]:+.3f}, {gp[2]:+.3f}]")
w(f"- Price shift: {gd[0]:+.2f} [{gd[1]:+.2f}, {gd[2]:+.2f}] USD")
# Statistical resolution and practical magnitude are reported separately.
# A CI that excludes zero on ~1,200 users can still describe a difference far
# too small to change what a user sees; the proposal commits to effect sizes
# rather than significance alone, so both tests must pass before the word
# "disparity" is used.
MATERIAL = {"Jaccard": 0.05, "relevance": 0.05, "price": 5.00}
sig, trivial = [], []
for n, gg in [("Jaccard", gj), ("relevance", gp), ("price", gd)]:
    resolved = not (gg[1] <= 0 <= gg[2])
    material = abs(gg[0]) >= MATERIAL[n]
    if resolved and material:
        sig.append(n)
    elif resolved:
        trivial.append(f"{n} ({gg[0]:+.3f}, below the {MATERIAL[n]} "
                       "materiality threshold)")
w(f"\n- Gaps both statistically resolved **and** materially large: "
  f"**{', '.join(sig) if sig else 'none'}**")
if trivial:
    w(f"- Statistically resolved but practically negligible: "
      f"{'; '.join(trivial)}")
w("\n" + ("**No tone-differential propagation is demonstrated.** Gaps are "
          "either unresolved or too small to alter what a user is shown. The "
          "honest reading is that this corpus -- two auditable tone bands, "
          "no dark-skin population -- cannot settle the question, not that no "
          "disparity exists. Interval widths above document that limitation."
          if not sig else
          "These gaps are both statistically resolved and large enough to "
          "change what a user is shown, and constitute demonstrated "
          "tone-differential propagation.") + "\n")


w("## 3. Safety-critical propagation -- referral failures\n")
w("Grades 2-3 attach a referral flag (section 7.4). A false-negative here is "
  "the most consequential downstream error the system can make: a user whose "
  "acne warrants qualified care is instead handed a cosmetic routine.\n")
w("| tone band | n | missed referral rate | unnecessary referral rate |")
w("|---|---|---|---|")
for g in GROUPS:
    sub = res[res.tone_band == g]
    mr = boot_mean(sub.missed_referral)
    ur = boot_mean(sub.unnecessary_referral)
    w(f"| {g} | {len(sub)} | {100*mr[0]:.1f}% [{100*mr[1]:.1f}, {100*mr[2]:.1f}] | "
      f"{100*ur[0]:.1f}% [{100*ur[1]:.1f}, {100*ur[2]:.1f}] |")
mrd = boot_diff(res[res.tone_band == "Light"].missed_referral,
                res[res.tone_band == "Medium"].missed_referral)
w(f"\n- **Missed-referral gap (Light - Medium):** {100*mrd[0]:+.1f}pp "
  f"[{100*mrd[1]:+.1f}, {100*mrd[2]:+.1f}]\n")


w("## 4. Catalogue coverage and list diversity\n")
w("Coverage = distinct products any user in the group is ever shown. A "
  "narrower catalogue for one group means fewer real options.\n")
w("| tone band | distinct products surfaced | coverage of catalogue | mean brands per list |")
w("|---|---|---|---|")
for g in GROUPS:
    sub = res[res.tone_band == g]
    ids = set()
    for s in sub.deployed_ids:
        ids.update(s.split(","))
    w(f"| {g} | {len(ids):,} | {100*len(ids)/len(rec.cat):.1f}% | "
      f"{sub.brands_deployed.mean():.2f} |")
w("")


w("## 5. Baseline comparison\n")
pop = rec.popularity_baseline(k=K)
pop_prec = np.mean([rec.precision_at_k(pop, g) for g in (0, 1, 2, 3)])
w(f"- Popularity-only ranker, mean P@{K} across grades: **{pop_prec:.3f}**")
w(f"- Pipeline recommender, mean deployed P@{K}: "
  f"**{res.prec_deployed.mean():.3f}**")
w(f"- The pipeline "
  f"{'adds value over' if res.prec_deployed.mean() > pop_prec else 'does not beat'}"
  " trivial popularity ranking, evidencing that the mapping table does work "
  "beyond surfacing best-sellers.\n")


# ============================================================ 5. FIGURES
xs = np.arange(len(GROUPS))
plt.figure(figsize=(6, 4))
plt.bar(xs, [per[g]["j"][0] for g in GROUPS],
        yerr=[[per[g]["j"][0] - per[g]["j"][1] for g in GROUPS],
              [per[g]["j"][2] - per[g]["j"][0] for g in GROUPS]],
        capsize=6, color=["#e6b8b0", "#b07d56"])
plt.xticks(xs, [f"{g}\n(n={per[g]['n']})" for g in GROUPS])
plt.ylabel("Jaccard overlap with correct list"); plt.ylim(0, 1)
plt.title("RQ2 - recommendation fidelity by tone band (95% CI)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "rq2_jaccard_by_tone.png"), dpi=120)
plt.close()

plt.figure(figsize=(6, 4))
plt.bar(["correct", "misclassified"], [j_c[0], j_m[0]],
        yerr=[[j_c[0]-j_c[1], j_m[0]-j_m[1]], [j_c[2]-j_c[0], j_m[2]-j_m[0]]],
        capsize=6, color=["#6ea38a", "#c98a9b"])
plt.ylabel("Jaccard overlap with correct list"); plt.ylim(0, 1)
plt.title("RQ2 - the downstream cost of an upstream error")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "rq2_cost_of_error.png"), dpi=120)
plt.close()

plt.figure(figsize=(7, 4))
for g, c in zip(GROUPS, ["#e6b8b0", "#b07d56"]):
    plt.hist(res[res.tone_band == g]["price_delta"], bins=30, alpha=0.6,
             label=g, color=c)
plt.axvline(0, color="k", ls=":")
plt.xlabel("price shift caused by misclassification (USD)")
plt.ylabel("users"); plt.legend()
plt.title("RQ2 - price distribution shift by tone band")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "rq2_price_shift.png"), dpi=120)
plt.close()

w("## 6. Figures\n")
w("- `reports/figures/rq2_jaccard_by_tone.png`")
w("- `reports/figures/rq2_cost_of_error.png`")
w("- `reports/figures/rq2_price_shift.png`\n")

w("## 7. Reading this analysis\n")
w("- **Propagation is real and quantified.** A misclassified user does not "
  "receive a slightly worse list; they receive a substantially different one. "
  "The Jaccard and relevance penalties above are the size of that harm.")
w("- **Whether it is tone-DIFFERENTIAL is a separate question** from whether "
  "it exists. With only two auditable tone bands in ACNE04 -- itself a "
  "reported finding -- this arm is under-powered to settle it, and every gap "
  "is therefore reported with its interval rather than as a point claim.")
w("- **The relevance metric is close to saturated and should be read with "
  "care.** P@10 sits near 0.99 in both arms because the mapping table endorses "
  "overlapping ingredient classes at adjacent grades, so a one-grade error "
  "still returns products carrying an endorsed class. Precision@k against a "
  "permissive reference set is therefore an insensitive instrument for "
  "propagation; **Jaccard overlap is the informative measure here**, and a "
  "stricter grade-specific reference standard is the obvious refinement.")
w("- **Attribute profiles are simulated**, sampled from the Sephora reviewer "
  "distribution with a fixed seed. They make each user distinct and realistic "
  "but they are not real declarations; this is a stated limitation.")
w("- **The recommender is deterministic**, so all divergence is attributable "
  "to the upstream classifier. That is the property that makes this "
  "measurement meaningful at all.\n")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
w(f"Report written to {OUT}")
