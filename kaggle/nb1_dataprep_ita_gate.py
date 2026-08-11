# %% [markdown]
# # NB1 — Data prep, ITA skin-tone derivation, and the go/no-go gate
#
# **Project:** *Does Bias Propagate? Fairness in Automated Acne Severity
# Assessment and Downstream Skincare Recommendation.*
#
# This notebook is CPU-only and fast (minutes). Run it **first**. It:
#
# 1. Discovers the three datasets attached as Kaggle inputs.
# 2. Computes **Individual Typology Angle (ITA)** for every ACNE04 and
#    Fitzpatrick17k image, from non-lesional skin pixels in CIELAB space.
# 3. Bins ITA into ordered tone groups and **validates ITA against the
#    ground-truth Fitzpatrick labels** (this is the RQ6 label-validity check).
# 4. Runs the **§10.1 go/no-go gate**: are ACNE04's darker-tone cells dense
#    enough for a well-powered fairness audit?
# 5. Writes per-image tables + stratified splits to `/kaggle/working/` for
#    the training notebooks (NB2 Arm A, NB3 Arm B) to consume.
#
# **Attach these Kaggle datasets as inputs (Add Input):**
# - `lexuanhieu131297/acne-severity-classification`  (ACNE04 — Arm B)
# - `nazmussadat013/fitzpatrick-17k-dataset`         (Fitzpatrick17k — Arm A)
# - `nadyinky/sephora-products-and-skincare-reviews` (Sephora — recommender/RQ5)

# %%
import os
import glob
import json
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    import cv2
except ImportError:
    os.system("pip install opencv-python-headless -q")
    import cv2

import matplotlib
import matplotlib.pyplot as plt

WORK = "/kaggle/working"
INPUT = "/kaggle/input"
os.makedirs(WORK, exist_ok=True)
RNG = np.random.default_rng(42)
pd.set_option("display.width", 140)


# %% [markdown]
# ## 1. Discover the input datasets
# Kaggle mounts datasets read-only under `/kaggle/input/<slug>`. Folder layouts
# vary between mirrors, so we discover rather than assume. Inspect the printed
# tree; if a path differs, only the small `find_*` helpers below need editing.

# %%
def show_tree(root, max_depth=3, max_per_dir=8):
    root = root.rstrip("/")
    base = root.count("/")
    for dirpath, dirs, files in os.walk(root):
        depth = dirpath.count("/") - base
        if depth > max_depth:
            dirs[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/  "
              f"({len(files)} files, {len(dirs)} subdirs)")
        for f in files[:max_per_dir]:
            print(f"{indent}  {f}")
        if len(files) > max_per_dir:
            print(f"{indent}  ... (+{len(files) - max_per_dir} more)")


for d in sorted(glob.glob(os.path.join(INPUT, "*"))):
    print("=" * 70)
    print("INPUT:", d)
    show_tree(d, max_depth=3)


# %%
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def all_images(root):
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(IMG_EXT):
                out.append(os.path.join(dp, f))
    return out


def _img_count(root):
    ext = (".jpg", ".jpeg", ".png", ".bmp")
    return sum(1 for _, _, fs in os.walk(root)
               for f in fs if f.lower().endswith(ext))


def find_dataset_dir(keywords):
    """Return the matching dir ANYWHERE under /kaggle/input holding the MOST
    images. Robust to nested mounts and to multiple mirrors of the same name
    (e.g. a 13-image stub vs the full set) being attached at once."""
    best, best_n = None, -1
    for dp, _, _ in os.walk(INPUT):
        if any(k in os.path.basename(dp).lower() for k in keywords):
            n = _img_count(dp)
            if n > best_n:
                best, best_n = dp, n
    return best


ACNE_DIR = find_dataset_dir(["acne"])
FITZ_DIR = find_dataset_dir(["fitz"])
SEPH_DIR = find_dataset_dir(["sephora"])
print("ACNE04     ->", ACNE_DIR)
print("Fitzpatrick->", FITZ_DIR)
print("Sephora    ->", SEPH_DIR)


# %% [markdown]
# ## 2. ITA — Individual Typology Angle
#
# ITA = arctan((L\* − 50) / b\*) · 180/π, computed on **non-lesional skin**
# pixels in CIELAB. Higher ITA ⇒ lighter skin. We approximate "non-lesional
# skin" without manual masks by:
# 1. a permissive skin-colour mask in YCrCb **and** HSV,
# 2. discarding the darkest/brightest pixels (shadow, specular highlight,
#    hair, lesions) via robust percentile trimming,
# 3. taking the **median** L\* and b\* of survivors (robust to outliers).
#
# This is an objective *proxy*, not ground truth — a documented limitation
# (proposal §5.4). Illumination confounds it; §10.2 handles this via
# sensitivity analysis on bin boundaries (implemented below).

# %%
def skin_mask(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    m_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m_hsv = ((h <= 25) | (h >= 165)) & (s >= 25) & (s <= 230) & (v >= 40)
    return m_ycrcb & m_hsv


def compute_ita(path):
    """Return (ITA_degrees, L_median, b_median, skin_frac) or NaNs on failure."""
    img = cv2.imread(path)
    if img is None:
        return (np.nan, np.nan, np.nan, 0.0)
    # cap size for speed
    h, w = img.shape[:2]
    scale = 512.0 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    mask = skin_mask(img)
    frac = float(mask.mean())
    if mask.sum() < 200:                      # too little skin found
        return (np.nan, np.nan, np.nan, frac)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV LAB: L in [0,255], a,b in [0,255] offset by 128 -> convert
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0
    Ls, bs = L[mask], b[mask]
    # trim shadow/highlight extremes on luminance
    lo, hi = np.percentile(Ls, [10, 90])
    keep = (Ls >= lo) & (Ls <= hi)
    if keep.sum() < 100:
        keep = np.ones_like(Ls, dtype=bool)
    Lm, bm = float(np.median(Ls[keep])), float(np.median(bs[keep]))
    ita = math.degrees(math.atan2(Lm - 50.0, bm)) if bm != 0 else np.nan
    return (ita, Lm, bm, frac)


# ITA -> tone group (standard dermatology cut points), then coarse band
def ita_to_group(ita):
    if np.isnan(ita):
        return "unknown"
    if ita > 55:
        return "very_light"      # ~Fitz I
    if ita > 41:
        return "light"           # ~Fitz II
    if ita > 28:
        return "intermediate"    # ~Fitz III
    if ita > 19:
        return "tan"             # ~Fitz IV
    if ita > 10:
        return "brown"           # ~Fitz V
    return "dark"                # ~Fitz VI


GROUP_TO_BAND = {
    "very_light": "Light", "light": "Light",
    "intermediate": "Medium", "tan": "Medium",
    "brown": "Deep", "dark": "Deep",
}
BAND_ORDER = ["Light", "Medium", "Deep"]


def ita_table(paths, label_fn=None, limit=None):
    rows = []
    if limit:
        paths = paths[:limit]
    n = len(paths)
    for i, p in enumerate(paths):
        ita, L, b, frac = compute_ita(p)
        grp = ita_to_group(ita)
        rows.append(dict(
            path=p, filename=os.path.basename(p),
            ita=ita, L=L, b=b, skin_frac=frac,
            tone_group=grp, tone_band=GROUP_TO_BAND.get(grp, "unknown"),
            label=label_fn(p) if label_fn else None,
        ))
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{n}")
    return pd.DataFrame(rows)


# %% [markdown]
# ## 3. ACNE04 — load, derive severity label, compute ITA (Arm B)
#
# ACNE04 encodes Hayashi severity in the file path/name (original files use the
# `levle0..levle3` convention; mirrors sometimes use folder names `0..3` or
# `mild/moderate/severe/very_severe`). The `acne_label` helper covers all three;
# check the printed value counts and adjust if your mirror differs.

# %%
def acne_label(path):
    s = path.lower().replace("\\", "/")
    import re
    m = re.search(r"lev[le]{0,2}\s*([0-3])", s)        # levle0 / level0 / lev0
    if m:
        return int(m.group(1))
    # folder-name based
    for k, v in {"very_severe": 3, "very severe": 3, "severe": 2,
                 "moderate": 1, "mild": 0, "clear": 0}.items():
        if k in s:
            return v
    m = re.search(r"/([0-3])/", s)                     # /2/ style class folder
    if m:
        return int(m.group(1))
    return -1


acne_paths = all_images(ACNE_DIR) if ACNE_DIR else []
print(f"ACNE04 images found: {len(acne_paths)}")
if acne_paths:
    print("sample paths:")
    for p in acne_paths[:5]:
        print("  ", p.replace(INPUT, ""))

acne = ita_table(acne_paths, label_fn=acne_label)
print("\nSeverity label distribution (-1 = unparsed):")
print(acne["label"].value_counts().sort_index())
print(f"\nITA computed for {acne['ita'].notna().sum()}/{len(acne)} images "
      f"({acne['ita'].isna().mean()*100:.1f}% failed skin detection)")


# %% [markdown]
# ## 4. Fitzpatrick17k — load labels, compute ITA (Arm A)
#
# Fitzpatrick17k ships a metadata CSV with ground-truth `fitzpatrick` type
# (1–6, sometimes `fitzpatrick_scale`) and a condition label. We join it to the
# image files by filename/md5, compute ITA on the same images, and later
# validate ITA-derived groups against the ground-truth labels.

# %%
def load_fitz_meta(fitz_dir):
    csvs = glob.glob(os.path.join(fitz_dir, "**", "*.csv"), recursive=True)
    print("Fitzpatrick CSVs:", [c.replace(INPUT, "") for c in csvs])
    if not csvs:
        return None
    # pick the biggest CSV (the metadata table)
    csvs.sort(key=lambda c: os.path.getsize(c), reverse=True)
    df = pd.read_csv(csvs[0])
    print("meta columns:", list(df.columns))
    return df


fitz_meta = load_fitz_meta(FITZ_DIR) if FITZ_DIR else None
fitz_paths = all_images(FITZ_DIR) if FITZ_DIR else []
print(f"\nFitzpatrick images found: {len(fitz_paths)}")

# Compute ITA on a capped sample for speed on first run; raise/remove LIMIT
# for the full audit (the training notebook uses the full set regardless).
FITZ_LIMIT = int(os.environ.get("FITZ_LIMIT", "4000"))
fitz = ita_table(fitz_paths, limit=FITZ_LIMIT)
print(f"ITA computed for {fitz['ita'].notna().sum()}/{len(fitz)} sampled images")


# %%
# Attach ground-truth Fitzpatrick label to the ITA table by filename stem.
def attach_fitz_label(fitz_df, meta):
    if meta is None:
        fitz_df["fitz_type"] = np.nan
        fitz_df["condition"] = np.nan
        return fitz_df
    cols = {c.lower(): c for c in meta.columns}
    type_col = next((cols[c] for c in cols
                     if "fitzpatrick" in c or c == "fitz"), None)
    cond_col = next((cols[c] for c in cols
                     if c in ("label", "condition", "nine_partition_label",
                              "three_partition_label", "diagnosis")), None)
    key_col = next((cols[c] for c in cols
                    if c in ("md5hash", "md5", "hash", "filename",
                             "image", "file", "path", "url")), None)
    print(f"Fitz meta: type={type_col}, condition={cond_col}, key={key_col}")
    if key_col is None:
        fitz_df["fitz_type"] = np.nan
        fitz_df["condition"] = np.nan
        return fitz_df

    def stem(x):
        x = str(x)
        x = x.split("/")[-1].split("\\")[-1]
        return os.path.splitext(x)[0]

    meta = meta.copy()
    meta["_key"] = meta[key_col].map(stem)
    fitz_df["_key"] = fitz_df["filename"].map(stem)
    keep = [c for c in [type_col, cond_col, "_key"] if c]
    merged = fitz_df.merge(meta[keep], on="_key", how="left")
    merged = merged.rename(columns={type_col: "fitz_type"} if type_col else {})
    if cond_col:
        merged = merged.rename(columns={cond_col: "condition"})
    else:
        merged["condition"] = np.nan
    if "fitz_type" not in merged:
        merged["fitz_type"] = np.nan
    return merged


fitz = attach_fitz_label(fitz, fitz_meta)
print("\nGround-truth Fitzpatrick type distribution:")
print(fitz["fitz_type"].value_counts(dropna=False).sort_index())


# %% [markdown]
# ## 5. Validate ITA vs ground-truth Fitzpatrick labels (RQ6 sub-check)
# If ITA is a usable proxy, ITA-derived groups should track the true
# Fitzpatrick types monotonically. We report a cross-tab and Spearman
# correlation between derived tone rank and true type.

# %%
from scipy.stats import spearmanr

GROUP_RANK = {"very_light": 1, "light": 2, "intermediate": 3,
              "tan": 4, "brown": 5, "dark": 6}
val = fitz.dropna(subset=["fitz_type"]).copy()
val = val[val["tone_group"] != "unknown"]
if len(val) > 50:
    val["derived_rank"] = val["tone_group"].map(GROUP_RANK)
    val["fitz_num"] = pd.to_numeric(val["fitz_type"], errors="coerce")
    val = val.dropna(subset=["fitz_num"])
    val = val[val["fitz_num"].between(1, 6)]
    rho, p = spearmanr(val["derived_rank"], val["fitz_num"])
    print(f"Spearman(ITA-derived group rank, ground-truth Fitzpatrick) "
          f"= {rho:.3f}  (p={p:.2e}, n={len(val)})")
    print("\nCross-tab (rows = ITA group, cols = true Fitzpatrick type):")
    print(pd.crosstab(val["tone_group"], val["fitz_num"]))
else:
    print("Not enough labelled+ITA rows to validate; raise FITZ_LIMIT.")


# %% [markdown]
# ## 6. THE GO/NO-GO GATE (proposal §10.1)
#
# The tone range in ACNE04 is load-bearing for RQ1/RQ2. Decision rule
# (pre-committed): the primary applied-audit path proceeds only if the Deep
# band holds enough images for interval estimates — operationalised as
# **≥ 100 Deep-band images overall and ≥ 15 in each severity class present**.
# Otherwise the project pivots to weighting Arm A + RQ5, and the compressed
# range is itself reported as a finding.

# %%
GATE_MIN_DEEP_TOTAL = 100
GATE_MIN_DEEP_PER_CLASS = 15

a = acne[acne["ita"].notna()].copy()
band_counts = a["tone_band"].value_counts()
print("ACNE04 tone-band counts (ITA-derived):")
print(band_counts, "\n")

# proportion CIs (Wilson) on each band share
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (p, c - h, c + h)


ntot = len(a)
for band in BAND_ORDER:
    k = int(band_counts.get(band, 0))
    p, lo, hi = wilson(k, ntot)
    print(f"  {band:7s}: {k:4d}  share={p*100:5.1f}%  "
          f"95% CI [{lo*100:.1f}%, {hi*100:.1f}%]")

deep_total = int(band_counts.get("Deep", 0))
deep_by_class = (a[a["tone_band"] == "Deep"]
                 .query("label >= 0")["label"].value_counts().sort_index())
print("\nDeep-band images per severity class:")
print(deep_by_class)

print("\nSeverity x tone-band cross-tab (ITA-derived):")
print(pd.crosstab(a["label"], a["tone_band"]))

gate_pass = (deep_total >= GATE_MIN_DEEP_TOTAL and
             (len(deep_by_class) > 0 and
              deep_by_class.min() >= GATE_MIN_DEEP_PER_CLASS))
print("\n" + "=" * 60)
print(f"GO/NO-GO GATE: {'>>> GO (primary path)' if gate_pass else '>>> NO-GO (pivot)'}")
print(f"  Deep total = {deep_total} (need >= {GATE_MIN_DEEP_TOTAL})")
if len(deep_by_class):
    print(f"  Min Deep-per-class = {int(deep_by_class.min())} "
          f"(need >= {GATE_MIN_DEEP_PER_CLASS})")
print("  Interpretation: if NO-GO, ACNE04's compressed tone range is REPORTED")
print("  AS A FINDING; Arm A (Fitzpatrick) becomes the primary powered audit")
print("  and RQ5 carries extra analytical weight. This is expected and planned.")
print("=" * 60)


# %% [markdown]
# ## 7. Sensitivity analysis — do tone bands survive shifted ITA cut points?
# The §10.2 mitigation for illumination-confounded ITA: re-bin under ±3°
# shifted boundaries and check the band counts are stable in ordering.

# %%
def ita_to_band_shifted(ita, shift):
    if np.isnan(ita):
        return "unknown"
    cuts = [55 + shift, 41 + shift, 28 + shift, 19 + shift, 10 + shift]
    if ita > cuts[0]:
        return "Light"
    if ita > cuts[1]:
        return "Light"
    if ita > cuts[2]:
        return "Medium"
    if ita > cuts[3]:
        return "Medium"
    if ita > cuts[4]:
        return "Deep"
    return "Deep"


for shift in (-3, 0, 3):
    bc = a["ita"].map(lambda x: ita_to_band_shifted(x, shift)).value_counts()
    print(f"shift {shift:+d}°:  " +
          "  ".join(f"{b}={int(bc.get(b,0))}" for b in BAND_ORDER))


# %% [markdown]
# ## 8. Stratified splits + save artifacts for the training notebooks
# Splits are stratified jointly by outcome class and tone band so per-group
# evaluation is possible at test time. Files land in `/kaggle/working/`; add
# this notebook's output as an input to NB2/NB3 (or re-run inline).

# %%
from sklearn.model_selection import train_test_split


def make_split(df, class_col, seed=42, test=0.2, val=0.1):
    df = df.copy()
    df["strata"] = (df[class_col].astype(str) + "_" + df["tone_band"].astype(str))
    # collapse ultra-rare strata so the splitter doesn't choke
    vc = df["strata"].value_counts()
    rare = vc[vc < 3].index
    df.loc[df["strata"].isin(rare), "strata"] = "rare"
    train_val, test_df = train_test_split(
        df, test_size=test, random_state=seed,
        stratify=df["strata"] if df["strata"].nunique() > 1 else None)
    train_df, val_df = train_test_split(
        train_val, test_size=val / (1 - test), random_state=seed,
        stratify=train_val["strata"] if train_val["strata"].nunique() > 1 else None)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.loc[part.index, "split"] = name
    return df.drop(columns="strata")


acne_out = acne.copy()
acne_valid = acne_out[(acne_out["label"] >= 0)].copy()
if len(acne_valid) > 20:
    acne_valid = make_split(acne_valid, "label")
    acne_out = acne_out.merge(acne_valid[["path", "split"]], on="path", how="left")

acne_out.to_csv(os.path.join(WORK, "acne04_ita_labels.csv"), index=False)
fitz.to_csv(os.path.join(WORK, "fitz17k_ita_labels.csv"), index=False)

gate = dict(
    gate_pass=bool(gate_pass),
    deep_total=int(deep_total),
    deep_min_per_class=int(deep_by_class.min()) if len(deep_by_class) else 0,
    band_counts={b: int(band_counts.get(b, 0)) for b in BAND_ORDER},
    acne_n_ita=int(ntot),
    acne_n_total=int(len(acne)),
    thresholds=dict(min_deep_total=GATE_MIN_DEEP_TOTAL,
                    min_deep_per_class=GATE_MIN_DEEP_PER_CLASS),
)
with open(os.path.join(WORK, "gate_decision.json"), "w") as fh:
    json.dump(gate, fh, indent=2)
print("Saved:")
print("  /kaggle/working/acne04_ita_labels.csv")
print("  /kaggle/working/fitz17k_ita_labels.csv")
print("  /kaggle/working/gate_decision.json")
print(json.dumps(gate, indent=2))


# %% [markdown]
# ## 9. Figures
# %%
plt.figure(figsize=(6, 4))
a["ita"].plot(kind="hist", bins=40, color="#c98a9b")
for c in (10, 19, 28, 41, 55):
    plt.axvline(c, color="k", ls=":", lw=0.7)
plt.title("ACNE04 — ITA distribution (dashed = tone-group cut points)")
plt.xlabel("ITA (degrees)")
plt.tight_layout()
plt.savefig(os.path.join(WORK, "acne_ita_hist.png"), dpi=120)
plt.show()

plt.figure(figsize=(6, 4))
ct = pd.crosstab(a["label"], a["tone_band"])
ct.plot(kind="bar", stacked=True, ax=plt.gca(),
        color=["#e6b8b0", "#b07d56", "#5c3a26"])
plt.title("ACNE04 — severity class x tone band")
plt.xlabel("severity label"); plt.ylabel("images")
plt.tight_layout()
plt.savefig(os.path.join(WORK, "acne_severity_by_tone.png"), dpi=120)
plt.show()

print("\nNB1 complete. Next: run NB2 (Arm A / Fitzpatrick) and NB3 (Arm B / ACNE04).")
