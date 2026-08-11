# %% [markdown]
# # NB1b — ITA validation & method comparison (fixing the over-darkening)
#
# **Why this notebook exists.** NB1's first ITA pass labelled ACNE04 as ~59%
# **Deep**, ~36% Medium, ~4% Light. ACNE04 is a predominantly *Chinese* clinical
# corpus and should skew Light/Medium (Fitzpatrick III–IV). A 59%-Deep result is
# not credible — it means the estimator reads faces too dark, most likely because
# the skin mask captures shadow / hair and because acne inflammation and half-face
# lighting drag the luminance down (proposal §10.2: "ITA confounded by
# illumination").
#
# The proposal commits to validating ITA "by manual inspection of a stratified
# sample" and to "sensitivity analysis under alternative bin boundaries". This
# notebook does exactly that:
# 1. computes ITA under **three methods** of increasing selectivity,
# 2. compares the resulting tone-band distributions,
# 3. **renders montages of real face crops per predicted band** so the estimates
#    can be eyeballed,
# 4. recommends the method whose distribution and montages are credible,
# 5. writes a corrected per-image ITA table for the downstream fairness audit.
#
# CPU-only, ~a few minutes. **Attach input:**
# `lexuanhieu131297/acne-severity-classification`.

# %%
import os, glob, math, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try:
    import cv2
except ImportError:
    os.system("pip install opencv-python-headless -q"); import cv2
import matplotlib.pyplot as plt

INPUT, WORK = "/kaggle/input", "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def find_dir(keys):
    best, best_n = None, -1
    for dp, _, _ in os.walk(INPUT):
        if any(k in os.path.basename(dp).lower() for k in keys):
            n = sum(1 for _, _, fs in os.walk(dp)
                    for f in fs if f.lower().endswith(IMG_EXT))
            if n > best_n:
                best, best_n = dp, n
    return best


def acne_label(path):
    m = re.search(r"lev[le]{0,2}\s*([0-3])", path.lower())
    return int(m.group(1)) if m else -1


ACNE = find_dir(["acne"])
paths = [os.path.join(dp, f) for dp, _, fs in os.walk(ACNE)
         for f in fs if f.lower().endswith(IMG_EXT)]
print("ACNE04 images:", len(paths))


# %% [markdown]
# ## 1. Three ITA estimators
#
# All share the same colour maths — ITA = arctan((L\*−50)/b\*)·180/π on CIELAB —
# but differ in **which pixels count as representative healthy skin**:
#
# | Method | Pixel selection | Rationale |
# |---|---|---|
# | **A — current** | skin mask, trim L to 10–90 pct, median | NB1's original; vulnerable to shadow |
# | **B — bright, non-red** | skin mask, drop reddest 25% (acne), keep upper-half L, median | ITA is defined on *clear, well-lit* skin |
# | **C — central + bright** | central 60% crop, then method B | drops hair / background / neck |
#
# If the over-darkening is a shadow/lesion artefact, B and C should shift the
# distribution lighter and match the known demographic.

# %%
def _skin_mask(bgr):
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb); cr, cb = y[:, :, 1], y[:, :, 2]
    m1 = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m2 = ((h <= 25) | (h >= 165)) & (s >= 25) & (s <= 230) & (v >= 40)
    return m1 & m2


def _lab(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    return L, a, b


def _ita(L, b):
    return math.degrees(math.atan2(np.median(L) - 50.0, np.median(b))) \
        if len(L) and np.median(b) != 0 else np.nan


def _load(path, cap=512):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]; s = cap / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


def ita_A(path):
    img = _load(path)
    if img is None: return np.nan
    m = _skin_mask(img)
    if m.sum() < 200: return np.nan
    L, a, b = _lab(img); Ls, bs = L[m], b[m]
    lo, hi = np.percentile(Ls, [10, 90]); keep = (Ls >= lo) & (Ls <= hi)
    if keep.sum() < 100: keep = np.ones_like(Ls, bool)
    return _ita(Ls[keep], bs[keep])


def _bright_nonred(L, a, b, m):
    Ls, as_, bs = L[m], a[m], b[m]
    if len(Ls) < 100: return Ls, bs
    red_cut = np.percentile(as_, 75)          # drop reddest 25% (inflammation)
    keep = as_ <= red_cut
    Ls, bs = Ls[keep], bs[keep]
    if len(Ls) < 50: return L[m], b[m]
    lmed = np.median(Ls)                        # keep the brighter (well-lit) half
    keep2 = Ls >= lmed
    return Ls[keep2], bs[keep2]


def ita_B(path):
    img = _load(path)
    if img is None: return np.nan
    m = _skin_mask(img)
    if m.sum() < 200: return np.nan
    L, a, b = _lab(img)
    Ls, bs = _bright_nonred(L, a, b, m)
    return _ita(Ls, bs)


def ita_C(path):
    img = _load(path)
    if img is None: return np.nan
    h, w = img.shape[:2]
    y0, y1, x0, x1 = int(0.2 * h), int(0.8 * h), int(0.2 * w), int(0.8 * w)
    img = img[y0:y1, x0:x1]
    m = _skin_mask(img)
    if m.sum() < 200: return np.nan
    L, a, b = _lab(img)
    Ls, bs = _bright_nonred(L, a, b, m)
    return _ita(Ls, bs)


def band(ita):
    if np.isnan(ita): return "unknown"
    return "Light" if ita > 41 else "Medium" if ita > 19 else "Deep"


# %% [markdown]
# ## 2. Compute all three methods over the corpus
# %%
rows = []
for i, p in enumerate(paths):
    rows.append(dict(path=p, label=acne_label(p),
                     ita_A=ita_A(p), ita_B=ita_B(p), ita_C=ita_C(p)))
    if (i + 1) % 300 == 0: print(f"  ...{i+1}/{len(paths)}")
df = pd.DataFrame(rows)
for mth in ["A", "B", "C"]:
    df[f"band_{mth}"] = df[f"ita_{mth}"].map(band)

print("\nMedian ITA by method:")
print(df[["ita_A", "ita_B", "ita_C"]].median())
print("\nTone-band distribution by method (% of images):")
for mth in ["A", "B", "C"]:
    vc = df[f"band_{mth}"].value_counts(normalize=True).reindex(
        ["Light", "Medium", "Deep", "unknown"]).fillna(0) * 100
    print(f"  method {mth}:  " +
          "  ".join(f"{b}={vc[b]:.1f}%" for b in ["Light", "Medium", "Deep"]))


# %% [markdown]
# ## 3. Distribution comparison
# A credible estimator should move ACNE04 toward Medium/Light. Watch how much of
# the "Deep" mass in method A migrates under B and C.
# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
for ax, mth in zip(axes, ["A", "B", "C"]):
    ax.hist(df[f"ita_{mth}"].dropna(), bins=40, color="#8a6ea3")
    for c in (19, 41):
        ax.axvline(c, color="k", ls=":", lw=0.8)
    ax.set_title(f"Method {mth}  (median {df[f'ita_{mth}'].median():.1f}°)")
    ax.set_xlabel("ITA (deg)")
axes[0].set_ylabel("images")
plt.suptitle("ITA distribution by method — dashed lines = Deep|Medium|Light cuts")
plt.tight_layout()
plt.savefig(os.path.join(WORK, "ita_method_distributions.png"), dpi=120)
plt.show()


# %% [markdown]
# ## 4. THE MANUAL-INSPECTION MONTAGES (the actual validation)
# For each method, show random face crops from each predicted band with their ITA
# printed. **This is where you judge correctness:** the "Deep" column should
# contain genuinely dark-skinned faces — if it is full of shadowed or reddened
# light-skinned faces, that method is mis-estimating.
# %%
def montage(method, n_per_band=6, seed=0):
    rng = np.random.default_rng(seed)
    bands = ["Light", "Medium", "Deep"]
    fig, axes = plt.subplots(len(bands), n_per_band,
                             figsize=(2.1 * n_per_band, 2.4 * len(bands)))
    fig.suptitle(f"Method {method} — sample faces per predicted tone band",
                 fontsize=13)
    for r, bd in enumerate(bands):
        sub = df[df[f"band_{method}"] == bd]
        pick = sub.sample(min(n_per_band, len(sub)), random_state=int(rng.integers(1e6))) \
            if len(sub) else sub
        for c in range(n_per_band):
            ax = axes[r, c]; ax.axis("off")
            if c == 0:
                ax.set_ylabel(bd, rotation=0, labelpad=30,
                              fontsize=12, va="center")
            if c < len(pick):
                row = pick.iloc[c]
                img = _load(row["path"], cap=256)
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"ITA {row[f'ita_{method}']:.0f}°",
                                 fontsize=8)
    plt.tight_layout()
    out = os.path.join(WORK, f"ita_montage_method_{method}.png")
    plt.savefig(out, dpi=110); plt.show()
    print("saved", out)


for mth in ["A", "B", "C"]:
    montage(mth)


# %% [markdown]
# ## 5. Cross-method agreement + sensitivity
# How often do the methods agree on the band? Large disagreement = the estimate
# is fragile and any tone-based finding must carry that caveat.
# %%
print("Agreement A vs B:",
      (df["band_A"] == df["band_B"]).mean().round(3))
print("Agreement B vs C:",
      (df["band_B"] == df["band_C"]).mean().round(3))
print("\nBand cross-tab A (rows) vs C (cols):")
print(pd.crosstab(df["band_A"], df["band_C"]))

# sensitivity of the chosen method to +/-3 deg cut shifts
def band_shift(ita, s):
    if np.isnan(ita): return "unknown"
    return "Light" if ita > 41 + s else "Medium" if ita > 19 + s else "Deep"

print("\nSensitivity of method C to +/-3 deg cut shifts:")
for s in (-3, 0, 3):
    vc = df["ita_C"].map(lambda x: band_shift(x, s)).value_counts()
    print(f"  shift {s:+d}:  " +
          "  ".join(f"{b}={int(vc.get(b,0))}" for b in ["Light", "Medium", "Deep"]))


# %% [markdown]
# ## 6. Choose a method and save the corrected table
# **Set `CHOSEN` below after looking at the montages.** Default is C (most
# selective). The saved CSV replaces NB1's ITA labels for the fairness audit.
# %%
CHOSEN = "C"           # <-- change to "A"/"B" if its montage looks more correct
df["ita"] = df[f"ita_{CHOSEN}"]
df["tone_band"] = df[f"band_{CHOSEN}"]
out = df[["path", "label", "ita", "tone_band",
          "ita_A", "ita_B", "ita_C"]].copy()
out.to_csv(os.path.join(WORK, "acne04_ita_validated.csv"), index=False)

print(f"CHOSEN method = {CHOSEN}")
print("Final tone-band counts:\n", out["tone_band"].value_counts())
print("\nFinal severity x tone-band:\n",
      pd.crosstab(out["label"], out["tone_band"]))
print("\nSaved /kaggle/working/acne04_ita_validated.csv")
print("\nJUDGEMENT NEEDED: open the three montage PNGs. Pick the method whose")
print("'Deep' row holds genuinely dark-skinned faces (not shadowed/reddened")
print("light skin). Set CHOSEN to it, re-run this last cell, and use the CSV.")
