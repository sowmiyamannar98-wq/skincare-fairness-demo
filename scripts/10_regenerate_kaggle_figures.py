"""
10_regenerate_kaggle_figures.py
------------------------------------------------------------------
Regenerates, locally, the figures that were produced inside Kaggle notebooks,
using the reported summary statistics rather than the raw run artefacts.

Why this exists: Kaggle wipes /kaggle/working when an interactive session ends,
and several notebooks were run interactively. The underlying numbers are all
recorded in reports/10, /11, /12 and /13, so the figures can be rebuilt without
re-running any GPU job.

Figures produced:
  rq4_tone_sensitivity.png   RQ4 flip rate + directional bias    (reports/10)
  rq3_tradeoff_curve.png     RQ3 accuracy-fairness trade-off     (reports/11)
  arch_comparison.png        architecture comparison             (reports/12)
  sentiment_by_tone.png      sentiment accuracy by tone          (reports/13)

NOT reproducible here (needs the source images / raw per-row data):
  ita_montage_method_{A,B,C}.png   rendered from actual ACNE04 faces
  ita_method_distributions.png     needs per-image ITA values
  rq4_shift_example.png            rendered from an actual face
  sentiment scatter panel          needs per-product scores (2,255 rows)

Run:  python scripts/10_regenerate_kaggle_figures.py
------------------------------------------------------------------
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join("reports", "figures")
os.makedirs(FIG, exist_ok=True)


def save(name):
    p = os.path.join(FIG, name)
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {p}")


# ============================================================ RQ4
# Source: reports/10_rq4_counterfactual.md
shifts = np.array([-20, -10, 0, 10, 20])
flip = np.array([6.8, 3.6, 0.0, 5.4, 20.1])
flip_lo = np.array([5.5, 2.6, 0.0, 4.2, 17.8])
flip_hi = np.array([8.2, 4.5, 0.0, 6.6, 22.0])
dexp = np.array([0.0556, 0.0193, 0.0000, -0.0049, 0.1036])
dexp_lo = np.array([0.0461, 0.0134, 0.0000, -0.0141, 0.0815])
dexp_hi = np.array([0.0652, 0.0261, 0.0000, 0.0047, 0.1261])

print("RQ4 ...")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].errorbar(shifts, flip, yerr=[flip - flip_lo, flip_hi - flip],
               fmt="o-", capsize=5, color="#8a6ea3", lw=1.8, markersize=7)
ax[0].axvline(0, color="grey", ls=":", lw=0.8)
ax[0].set_xlabel("ΔITA (degrees)")
ax[0].set_ylabel("% of predictions flipped")
ax[0].set_title("Prediction instability under tone shift")
ax[0].grid(alpha=0.25)

ax[1].errorbar(shifts, dexp, yerr=[dexp - dexp_lo, dexp_hi - dexp],
               fmt="s-", capsize=5, color="#c98a9b", lw=1.8, markersize=7)
ax[1].axhline(0, color="k", ls=":", lw=0.9)
ax[1].set_xlabel("ΔITA (degrees)")
ax[1].set_ylabel("Δ expected severity")
ax[1].set_title("Directional tone bias")
ax[1].grid(alpha=0.25)
fig.suptitle("RQ4 — counterfactual tone probe (lesion content held constant)",
             fontsize=12)
save("rq4_tone_sensitivity.png")


# ============================================================ RQ3
# Source: reports/11_rq3_mitigation.md
strat = ["group_reweight", "erm", "class_reweight", "oversample",
         "group_dro", "focal"]
f1 = np.array([0.743, 0.735, 0.731, 0.731, 0.726, 0.708])
f1sd = np.array([0.035, 0.031, 0.026, 0.035, 0.011, 0.036])
accgap = np.array([0.034, 0.036, 0.052, 0.082, 0.033, 0.070])
accgap_sd = np.array([0.021, 0.021, 0.013, 0.012, 0.017, 0.022])
ecegap = np.array([0.051, 0.027, 0.038, 0.034, 0.014, 0.071])
ecegap_sd = np.array([0.023, 0.023, 0.009, 0.009, 0.016, 0.051])

print("RQ3 ...")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, gap, gsd, name in [
        (ax[0], accgap, accgap_sd, "|accuracy gap| (Light − Medium)"),
        (ax[1], ecegap, ecegap_sd, "|calibration gap| (ECE)")]:
    for i, s in enumerate(strat):
        marker = "*" if s == "erm" else "o"
        size = 16 if s == "erm" else 10
        a.errorbar(gap[i], f1[i], xerr=gsd[i], yerr=f1sd[i], fmt=marker,
                   capsize=4, markersize=size, color="#8a6ea3")
        a.annotate(s, (gap[i], f1[i]), textcoords="offset points",
                   xytext=(8, 6), fontsize=8.5)
    a.set_xlabel(name)
    a.set_ylabel("macro-F1")
    a.set_title(f"macro-F1 vs {name}")
    a.grid(alpha=0.25)
fig.suptitle("RQ3 — mitigation trade-off (error bars = seed sd; ★ = unmitigated "
             "baseline).\nAll strategies overlap: no strategy is resolved "
             "against the baseline.", fontsize=11)
save("rq3_tradeoff_curve.png")


# ============================================================ ARCHITECTURE
# Source: reports/12_architecture_comparison.md
arch = ["resnet50", "efficientnet_b0", "efficientnet_b3", "vit_base"]
params = np.array([23.5, 4.0, 10.7, 85.8])
af1 = np.array([0.731, 0.690, 0.679, 0.639])
af1sd = np.array([0.026, 0.039, 0.030, 0.038])
pooled_sd = 0.033

print("architecture ...")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
x = np.arange(len(arch))
bars = ax[0].bar(x, af1, yerr=af1sd, capsize=6,
                 color=["#6ea38a", "#8a6ea3", "#8a6ea3", "#c98a9b"])
ax[0].axhline(af1.max() - pooled_sd, color="k", ls=":", lw=1,
              label=f"best − 1 seed sd ({pooled_sd:.3f})")
ax[0].set_xticks(x)
ax[0].set_xticklabels(arch, rotation=18, ha="right")
ax[0].set_ylabel("macro-F1")
ax[0].set_ylim(0.55, 0.80)
ax[0].legend(fontsize=8)
ax[0].set_title("Architecture comparison (error bars = seed sd)")
ax[0].grid(alpha=0.25, axis="y")

ax[1].errorbar(params, af1, yerr=af1sd, fmt="o", capsize=4, markersize=10,
               color="#c98a9b", lw=0)
for i, a_ in enumerate(arch):
    ax[1].annotate(a_, (params[i], af1[i]), textcoords="offset points",
                   xytext=(9, 5), fontsize=8.5)
ax[1].set_xlabel("parameters (millions)")
ax[1].set_ylabel("macro-F1")
ax[1].set_title("Capacity vs performance — more parameters do not help")
ax[1].grid(alpha=0.25)
save("arch_comparison.png")


# ============================================================ SENTIMENT
# Source: reports/13_sentiment_model.md
bands = ["Light", "Medium", "Deep"]
n = [15887, 3630, 575]
acc = np.array([0.9751, 0.9774, 0.9704])
lo = np.array([0.9726, 0.9725, 0.9565])
hi = np.array([0.9773, 0.9821, 0.9827])

print("sentiment ...")
plt.figure(figsize=(6.5, 4.4))
plt.bar(bands, acc, yerr=[acc - lo, hi - acc], capsize=7,
        color=["#e6b8b0", "#b07d56", "#5c3a26"])
for i, b in enumerate(bands):
    plt.text(i, lo[i] - 0.004, f"n={n[i]:,}", ha="center", fontsize=8.5,
             color="white" if i == 2 else "black")
plt.ylim(0.94, 1.0)
plt.ylabel("accuracy")
plt.title("Sentiment accuracy by reviewer tone band (95% CI)\n"
          "spread 0.0070 — no disparity; note interval width tracks sample size",
          fontsize=10.5)
plt.grid(alpha=0.25, axis="y")
save("sentiment_by_tone.png")

print("\nDone. Still needed from Kaggle (cannot be rebuilt locally):")
print("  ita_montage_method_A/B/C.png   — rendered from ACNE04 faces")
print("  ita_method_distributions.png   — needs per-image ITA values")
print("  rq4_shift_example.png          — rendered from an ACNE04 face")
