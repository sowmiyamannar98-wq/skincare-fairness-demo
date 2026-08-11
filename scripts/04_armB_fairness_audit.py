"""
04_armB_fairness_audit.py
------------------------------------------------------------------
RQ1 (applied arm) — disaggregated fairness audit of the Arm B acne-severity
classifier, using the VALIDATED (Method B) ITA tone labels from NB1b.

Inputs (download from the Kaggle notebook outputs, drop into data/kaggle_out/):
  data/kaggle_out/armB_oof_predictions.csv   (from NB3)
  data/kaggle_out/acne04_ita_validated.csv   (from NB1b)

Why Method B: NB1b showed the original ITA over-darkened (59% "Deep" on a
Chinese-majority corpus). Method B's median ITA = 37 deg (Fitzpatrick III),
matching the demographic. Montage inspection showed the residual "Deep" bucket
is lighting/acne artifact, not genuinely dark skin -> the audit is therefore
restricted to Light vs Medium, and Deep is reported as unauditable (a finding).

Run:  python scripts/04_armB_fairness_audit.py

Outputs:
  reports/04_armB_fairness_audit.md
  reports/figures/armB_fair_*.png
------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score

KO = os.path.join("data", "kaggle_out")
FIG = os.path.join("reports", "figures")
OUT = os.path.join("reports", "04_armB_fairness_audit.md")
os.makedirs(FIG, exist_ok=True)

PRED = os.path.join(KO, "armB_oof_predictions.csv")
ITA = os.path.join(KO, "acne04_ita_validated.csv")
TONE_METHOD = "ita_B"          # primary; ita_C available as sensitivity
GROUPS = ["Light", "Medium"]   # Deep excluded — see module docstring
B = 2000                       # bootstrap resamples
RNG = np.random.default_rng(42)
lines = []


def w(s=""):
    try:
        print(s)
    except UnicodeEncodeError:                 # Windows cp1252 console
        print(s.encode("ascii", "replace").decode("ascii"))
    lines.append(s)                            # file is written UTF-8


def stem(p):
    p = str(p).replace("\\", "/").split("/")[-1]
    return os.path.splitext(p)[0]


def band_from_ita(ita):
    if pd.isna(ita):
        return "unknown"
    return "Light" if ita > 41 else "Medium" if ita > 19 else "Deep"


# ---------------------------------------------------------------- load + join
if not (os.path.exists(PRED) and os.path.exists(ITA)):
    raise SystemExit(
        f"Missing inputs. Put the two Kaggle CSVs in {KO}/:\n"
        f"  - armB_oof_predictions.csv (NB3)\n"
        f"  - acne04_ita_validated.csv (NB1b)")

pred = pd.read_csv(PRED)
ita = pd.read_csv(ITA)
# the OOF file carries the OLD (broken) tone_band from NB3; drop it so the
# validated Method-B band from NB1b is the one used.
pred = pred.drop(columns=[c for c in ["tone_band"] if c in pred.columns])
pred["key"] = pred["path"].map(stem)
ita["key"] = ita["path"].map(stem)
ita["tone_band"] = ita[TONE_METHOD].map(band_from_ita)   # re-band from Method B
df = pred.merge(ita[["key", TONE_METHOD, "tone_band"]], on="key", how="left")

w("# RQ1 (Arm B) — Disaggregated Fairness Audit of the Acne-Severity Classifier\n")
w(f"_Tone labels: validated ITA **{TONE_METHOD}** (Method B, NB1b). "
  "Audit restricted to Light vs Medium; Deep excluded as artifact-laden "
  "(NB1b montage evidence)._\n")
w(f"- Predictions joined: **{df['tone_band'].notna().sum()} / {len(pred)}**")
w("- Tone-band counts (validated):")
for b, n in df["tone_band"].value_counts().items():
    w(f"    - {b}: {n}")
w("")

overall_acc = accuracy_score(df["y_true"], df["y_pred"])
overall_f1 = f1_score(df["y_true"], df["y_pred"], average="macro")
w(f"- **Overall** (all groups): accuracy **{overall_acc:.3f}**, "
  f"macro-F1 **{overall_f1:.3f}**\n")


# ---------------------------------------------------------------- bootstrap
def boot_metric(y_true, y_pred, metric, b=B):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    point = metric(y_true, y_pred)
    vals = np.empty(b)
    for i in range(b):
        idx = RNG.integers(0, n, n)
        vals[i] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (point, lo, hi)


def acc_m(yt, yp):
    return accuracy_score(yt, yp)


def f1_m(yt, yp):
    return f1_score(yt, yp, average="macro")


def ece(conf, correct, n_bins=10):
    """Expected Calibration Error."""
    conf = np.asarray(conf)
    correct = np.asarray(correct).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e, N = 0.0, len(conf)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return e


# ---------------------------------------------------------------- per group
w("## 1. Per-group performance (bootstrapped 95% CIs)\n")
w("| group | n | accuracy | macro-F1 | ECE |")
w("|---|---|---|---|---|")
per = {}
for g in GROUPS:
    sub = df[df["tone_band"] == g]
    a = boot_metric(sub["y_true"], sub["y_pred"], acc_m)
    f = boot_metric(sub["y_true"], sub["y_pred"], f1_m)
    correct = (sub["y_true"].values == sub["y_pred"].values)
    e = ece(sub["confidence"].values, correct) if "confidence" in sub else np.nan
    per[g] = dict(n=len(sub), acc=a, f1=f, ece=e)
    w(f"| {g} | {len(sub)} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | "
      f"{f[0]:.3f} [{f[1]:.3f}, {f[2]:.3f}] | {e:.3f} |")
w("")


# ---------------------------------------------------------------- gaps
def boot_gap(gA, gB, metric, b=B):
    """Bootstrap CI for metric(A) - metric(B)."""
    a = df[df["tone_band"] == gA]
    c = df[df["tone_band"] == gB]
    ya, pa = a["y_true"].values, a["y_pred"].values
    yc, pc = c["y_true"].values, c["y_pred"].values
    if len(ya) == 0 or len(yc) == 0:
        return (np.nan, np.nan, np.nan)
    point = metric(ya, pa) - metric(yc, pc)
    vals = np.empty(b)
    for i in range(b):
        ia = RNG.integers(0, len(ya), len(ya))
        ic = RNG.integers(0, len(yc), len(yc))
        vals[i] = metric(ya[ia], pa[ia]) - metric(yc[ic], pc[ic])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (point, lo, hi)


w("## 2. Fairness gaps (Light − Medium)\n")
ga = boot_gap("Light", "Medium", acc_m)
gf = boot_gap("Light", "Medium", f1_m)
w(f"- **Accuracy gap:** {ga[0]:+.3f} [95% CI {ga[1]:+.3f}, {ga[2]:+.3f}]")
w(f"- **Macro-F1 gap:** {gf[0]:+.3f} [95% CI {gf[1]:+.3f}, {gf[2]:+.3f}]")
sig = "does NOT" if (ga[1] <= 0 <= ga[2]) else "DOES"
w(f"- The accuracy-gap CI {sig} exclude zero → disparity is "
  f"{'not statistically resolved' if sig=='does NOT' else 'statistically supported'} "
  "at this sample size.\n")

# equal opportunity: per-class recall (TPR) parity across groups
w("## 3. Equal-opportunity view — per-class recall by group\n")
classes = sorted(df["y_true"].unique())
w("| severity class | recall Light | recall Medium | gap |")
w("|---|---|---|---|")
for c in classes:
    rl = recall_score(df[df.tone_band == "Light"]["y_true"] == c,
                      df[df.tone_band == "Light"]["y_pred"] == c,
                      zero_division=0)
    rm = recall_score(df[df.tone_band == "Medium"]["y_true"] == c,
                      df[df.tone_band == "Medium"]["y_pred"] == c,
                      zero_division=0)
    w(f"| {c} | {rl:.3f} | {rm:.3f} | {rl-rm:+.3f} |")
w("")


# ---------------------------------------------------------------- figures
xs = np.arange(len(GROUPS))
plt.figure(figsize=(6, 4))
plt.bar(xs, [per[g]["acc"][0] for g in GROUPS],
        yerr=[[per[g]["acc"][0] - per[g]["acc"][1] for g in GROUPS],
              [per[g]["acc"][2] - per[g]["acc"][0] for g in GROUPS]],
        capsize=6, color=["#e6b8b0", "#b07d56"])
plt.xticks(xs, [f"{g}\n(n={per[g]['n']})" for g in GROUPS])
plt.ylabel("accuracy"); plt.ylim(0, 1)
plt.title("Arm B — accuracy by tone group (95% bootstrap CI)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "armB_fair_accuracy.png"), dpi=120)
plt.close()

plt.figure(figsize=(6, 4))
plt.bar(xs, [per[g]["ece"] for g in GROUPS],
        color=["#e6b8b0", "#b07d56"])
plt.xticks(xs, [f"{g}\n(n={per[g]['n']})" for g in GROUPS])
plt.ylabel("Expected Calibration Error")
plt.title("Arm B — calibration error by tone group (lower = better)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "armB_fair_ece.png"), dpi=120)
plt.close()

w("## 4. Figures\n")
w("- `reports/figures/armB_fair_accuracy.png`")
w("- `reports/figures/armB_fair_ece.png`\n")

w("## 5. Reading this audit\n")
w("- **Scope is Light vs Medium only.** ACNE04 has no auditable dark-skin "
  "population (NB1b): the apparent 'Deep' images were lighting/acne artifacts. "
  "This is itself a reported finding on the limits of facial acne corpora.")
w("- **Interpret via the CIs, not point estimates.** The Light group is small, "
  "so its interval is wide; a point 'gap' that the CI straddles zero is not a "
  "demonstrated disparity.")
w("- **Calibration matters as much as accuracy** — a group can match on accuracy "
  "yet be worse-calibrated (confidently wrong), which the per-group ECE exposes "
  "and which justifies the abstention threshold in the demonstrator.")
w("- **This is the applied arm (Arm B).** The powered, ground-truth-tone audit "
  "is Arm A (Fitzpatrick); RQ6 compares the two.\n")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
w(f"Report written to {OUT}")
