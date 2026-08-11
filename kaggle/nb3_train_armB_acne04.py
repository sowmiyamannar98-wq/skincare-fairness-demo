# %% [markdown]
# # NB3 — Arm B: applied acne-severity classifier on ACNE04
#
# **Project:** *Does Bias Propagate?* — Arm B is the **applied audit** on the
# model the demonstrator actually runs (proposal §5.1, §7.1). ACNE04 facial
# images, Hayashi severity 0–3. Tone is **ITA-derived** (no ground-truth tone
# in facial data), so results are read together with NB1's go/no-go gate.
#
# Because ACNE04 is small (~1,457 images), a single held-out split gives an
# unstable disaggregated estimate. This notebook therefore supports **repeated
# stratified K-fold** (§7.1): every image gets an out-of-fold prediction, so the
# fairness audit draws on the whole corpus, not a 292-image test slice.
#
# GPU notebook — enable GPU, *Save & Run All*.
#
# **Attach inputs:**
# - `lexuanhieu131297/acne-severity-classification`  (ACNE04 images)
# - *(optional)* NB1's output containing `acne04_ita_labels.csv`. If absent,
#   this notebook recomputes ITA inline.

# %%
import os, glob, json, math, time, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
try:
    import timm
except ImportError:
    os.system("pip install timm -q"); import timm
try:
    import cv2
except ImportError:
    os.system("pip install opencv-python-headless -q"); import cv2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

INPUT, WORK = "/kaggle/input", "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# >>> SMOKE_TEST: set True for a ~2-min pipeline check (1 epoch, 2 folds,
#     subsampled). Confirm it writes armB_oof_predictions.csv, then set back
#     to False and use Save & Run All for the real run. <<<
SMOKE_TEST = False

CFG = dict(model="resnet50", img_size=224, batch=32, epochs=18, lr=3e-4,
           weight_decay=1e-4, seed=42, num_workers=2,
           n_folds=5, n_repeats=1,      # raise n_repeats for repeated CV
           smoke_subsample=None)
if SMOKE_TEST:
    CFG["epochs"] = 1
    CFG["n_folds"] = 2
    CFG["smoke_subsample"] = 300
    print(">>> SMOKE_TEST ON — quick pipeline check, NOT a real run <<<")
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


# %% [markdown]
# ## 1. Build the ACNE04 table (path, severity label, ITA tone band)
# Prefer NB1's `acne04_ita_labels.csv`; otherwise compute ITA inline.
# %%
def find_file(name):
    hits = glob.glob(os.path.join(INPUT, "**", name), recursive=True)
    return hits[0] if hits else None

def find_dir(keys):
    ext = (".jpg", ".jpeg", ".png", ".bmp")
    best, best_n = None, -1
    for dp, _, _ in os.walk(INPUT):
        if any(k in os.path.basename(dp).lower() for k in keys):
            n = sum(1 for _, _, fs in os.walk(dp)
                    for f in fs if f.lower().endswith(ext))
            if n > best_n:
                best, best_n = dp, n
    return best

def acne_label(path):
    s = path.lower().replace("\\", "/")
    m = re.search(r"lev[le]{0,2}\s*([0-3])", s)
    if m: return int(m.group(1))
    for k, v in {"very_severe": 3, "very severe": 3, "severe": 2,
                 "moderate": 1, "mild": 0, "clear": 0}.items():
        if k in s: return v
    m = re.search(r"/([0-3])/", s)
    return int(m.group(1)) if m else -1

# ---- inline ITA (same method as NB1) ----
def skin_mask(bgr):
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb); cr, cb = y[:, :, 1], y[:, :, 2]
    m1 = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m2 = ((h <= 25) | (h >= 165)) & (s >= 25) & (s <= 230) & (v >= 40)
    return m1 & m2

def compute_ita(path):
    img = cv2.imread(path)
    if img is None: return np.nan
    hh, ww = img.shape[:2]; sc = 512.0 / max(hh, ww)
    if sc < 1: img = cv2.resize(img, (int(ww * sc), int(hh * sc)))
    mask = skin_mask(img)
    if mask.sum() < 200: return np.nan
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0); b = lab[:, :, 2] - 128.0
    Ls, bs = L[mask], b[mask]
    lo, hi = np.percentile(Ls, [10, 90]); keep = (Ls >= lo) & (Ls <= hi)
    if keep.sum() < 100: keep = np.ones_like(Ls, bool)
    Lm, bm = float(np.median(Ls[keep])), float(np.median(bs[keep]))
    return math.degrees(math.atan2(Lm - 50.0, bm)) if bm != 0 else np.nan

def ita_band(ita):
    if np.isnan(ita): return "unknown"
    return "Light" if ita > 41 else "Medium" if ita > 19 else "Deep"

csv = find_file("acne04_ita_labels.csv")
if csv:
    print("using NB1 label file:", csv)
    df = pd.read_csv(csv)
    if "tone_band" not in df or df["tone_band"].isna().all():
        df["tone_band"] = df["ita"].map(ita_band)
    if "label" not in df:
        df["label"] = df["path"].map(acne_label)
else:
    ACNE = find_dir(["acne"]); assert ACNE, "ACNE04 not attached"
    paths = [os.path.join(dp, f) for dp, _, fs in os.walk(ACNE)
             for f in fs if f.endswith(IMG_EXT)]
    print(f"computing ITA inline for {len(paths)} images ...")
    rows = []
    for i, p in enumerate(paths):
        ita = compute_ita(p)
        rows.append(dict(path=p, label=acne_label(p), ita=ita,
                         tone_band=ita_band(ita)))
        if (i + 1) % 300 == 0: print(f"  ...{i+1}/{len(paths)}")
    df = pd.DataFrame(rows)

df = df[df["label"].between(0, 3)].reset_index(drop=True)
if CFG["smoke_subsample"]:
    df = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), CFG["smoke_subsample"] // 4),
                           random_state=CFG["seed"])).reset_index(drop=True)
    print(f">>> SMOKE subsample -> {len(df)} images")
print("\nseverity distribution:\n", df["label"].value_counts().sort_index())
print("\ntone-band distribution:\n", df["tone_band"].value_counts())
print("\nseverity x tone:\n", pd.crosstab(df["label"], df["tone_band"]))
classes = sorted(df["label"].unique()); n_cls = len(classes)


# %% [markdown]
# ## 2. Transforms + dataset
# %%
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = T.Compose([T.Resize((CFG["img_size"], CFG["img_size"])),
                      T.RandomHorizontalFlip(), T.RandomRotation(10),
                      T.ColorJitter(brightness=0.05, contrast=0.05),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize((CFG["img_size"], CFG["img_size"])),
                     T.ToTensor(), T.Normalize(MEAN, STD)])

class AcneDS(Dataset):
    def __init__(self, frame, tf):
        self.p = frame["path"].values; self.y = frame["label"].values
        self.tone = frame["tone_band"].values; self.tf = tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        return (self.tf(Image.open(self.p[i]).convert("RGB")),
                int(self.y[i]), self.tone[i], self.p[i])

def collate(b):
    xs, ys, t, p = zip(*b)
    return torch.stack(xs), torch.tensor(ys), list(t), list(p)


# %% [markdown]
# ## 3. Repeated stratified K-fold — out-of-fold predictions for the whole set
# %%
def make_model():
    return timm.create_model(CFG["model"], pretrained=True,
                             num_classes=n_cls).to(DEVICE)

def train_one(tr_idx, va_idx, tag):
    tr, va = df.iloc[tr_idx], df.iloc[va_idx]
    tr_dl = DataLoader(AcneDS(tr, train_tf), batch_size=CFG["batch"],
                       shuffle=True, num_workers=CFG["num_workers"],
                       pin_memory=True, collate_fn=collate)
    va_dl = DataLoader(AcneDS(va, eval_tf), batch_size=CFG["batch"],
                       shuffle=False, num_workers=CFG["num_workers"],
                       pin_memory=True, collate_fn=collate)
    model = make_model()
    counts = tr["label"].value_counts().reindex(classes).fillna(1).values
    w = torch.tensor(counts.sum() / (n_cls * counts),
                     dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
    for ep in range(CFG["epochs"]):
        model.train()
        for x, y, _, _ in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    # out-of-fold predictions
    model.eval(); out = []
    with torch.no_grad():
        for x, y, tones, paths in va_dl:
            x = x.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
                prob = torch.softmax(model(x), 1).cpu().numpy()
            pred = prob.argmax(1)
            for j in range(len(paths)):
                r = dict(path=paths[j], tone_band=tones[j], fold=tag,
                         y_true=int(y[j]), y_pred=int(pred[j]),
                         confidence=float(prob[j].max()))
                for c in range(n_cls): r[f"prob_{c}"] = float(prob[j][c])
                out.append(r)
    return model, pd.DataFrame(out)

oof, best_model, best_f1 = [], None, -1
for rep in range(CFG["n_repeats"]):
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True,
                          random_state=CFG["seed"] + rep)
    for k, (tri, vai) in enumerate(skf.split(df, df["label"])):
        t0 = time.time()
        model, preds = train_one(tri, vai, f"r{rep}f{k}")
        f1 = f1_score(preds.y_true, preds.y_pred, average="macro")
        print(f"rep{rep} fold{k}: macroF1={f1:.4f}  ({time.time()-t0:.0f}s)")
        oof.append(preds)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(),
                        "classes": classes, "cfg": CFG},
                       os.path.join(WORK, "armB_acne_resnet50_best.pt"))

oof_df = pd.concat(oof, ignore_index=True)
oof_df.to_csv(os.path.join(WORK, "armB_oof_predictions.csv"), index=False)


# %% [markdown]
# ## 4. Headline results + per-tone fairness signal
# %%
acc = accuracy_score(oof_df.y_true, oof_df.y_pred)
f1m = f1_score(oof_df.y_true, oof_df.y_pred, average="macro")
print(f"ARM B out-of-fold: acc={acc:.4f}  macro-F1={f1m:.4f}")
print("\nPer-tone-band (read WITH NB1 gate — Deep n may be tiny):")
for b in ["Light", "Medium", "Deep", "unknown"]:
    g = oof_df[oof_df.tone_band == b]
    if len(g):
        print(f"  {b:8s} n={len(g):4d}  acc={accuracy_score(g.y_true, g.y_pred):.4f}"
              f"  macroF1={f1_score(g.y_true, g.y_pred, average='macro'):.4f}")
print("\n", classification_report(oof_df.y_true, oof_df.y_pred))

with open(os.path.join(WORK, "armB_summary.json"), "w") as fh:
    json.dump(dict(oof_acc=acc, oof_macro_f1=f1m, best_fold_f1=best_f1,
                   n=len(oof_df), classes=[int(c) for c in classes],
                   folds=CFG["n_folds"], repeats=CFG["n_repeats"]), fh, indent=2)
print("\nSaved: armB_acne_resnet50_best.pt, armB_oof_predictions.csv, "
      "armB_summary.json")
print("NB3 (Arm B) complete. Fairness audit consumes the OOF predictions CSV.")
