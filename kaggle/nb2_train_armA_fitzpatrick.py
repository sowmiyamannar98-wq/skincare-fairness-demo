# %% [markdown]
# # NB2 — Arm A: reference fairness classifier on Fitzpatrick17k
#
# **Project:** *Does Bias Propagate?* — Arm A is the **well-powered reference
# audit** (proposal §5.3, §7.1). Ground-truth Fitzpatrick tone labels I–VI,
# ~16.5k images. This model is **NEVER connected to the recommender** — it is a
# pure offline fairness experiment (hard scope boundary, §5.3).
#
# GPU notebook. Enable **GPU T4 x2 / P100** and run *Save & Run All* for the
# overnight run. Trains a ResNet-50 baseline to classify skin condition
# (coarse 3-way partition), then saves **test-set predictions with tone labels**
# so the fairness audit (NB4 / local) can compute per-group metrics + CIs.
#
# **Attach input:** `nazmussadat013/fitzpatrick-17k-dataset`

# %%
import os, glob, json, math, time, warnings
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

INPUT = "/kaggle/input"
WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__)

# ---- config (edit here) -------------------------------------------------
# >>> SMOKE_TEST: set True for a ~2-min pipeline check (1 epoch, 400 imgs).
#     Confirm it trains + writes armA_test_predictions.csv, then set back to
#     False and use Save & Run All for the real overnight run. <<<
SMOKE_TEST = False

CFG = dict(
    model="resnet50", img_size=224, batch=64, epochs=20, lr=3e-4,
    weight_decay=1e-4, seed=42, num_workers=2,
    max_images=None,        # full dataset
)
if SMOKE_TEST:
    CFG["epochs"] = 1
    CFG["max_images"] = 400
    print(">>> SMOKE_TEST ON — quick pipeline check, NOT a real run <<<")
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])


# %% [markdown]
# ## 1. Locate images + metadata, build the label table
# Fitzpatrick17k metadata carries `fitzpatrick` (1–6) and partition labels.
# Target = `three_partition_label` (non-neoplastic / benign / malignant) —
# coarse and tractable, per §5.3. Tone band from ground-truth type:
# Light(1–2) / Medium(3–4) / Deep(5–6).

# %%
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

# matches fitzpatrick mirrors AND the BTTAI/AJL dermatology competition folder
FITZ = find_dir(["fitz", "bttai", "ajl", "dermatology", "equitable"])
assert FITZ, "Fitzpatrick/BTTAI input not attached"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")

img_paths = [os.path.join(dp, f) for dp, _, fs in os.walk(FITZ)
             for f in fs if f.endswith(IMG_EXT)]
print("images on disk:", len(img_paths))

# search the WHOLE input tree for a CSV (some mirrors keep it beside, not
# inside, the image folder) and pick the largest.
csvs = sorted(glob.glob(os.path.join(INPUT, "**", "*.csv"), recursive=True),
              key=os.path.getsize, reverse=True)
assert csvs, (
    "NO metadata CSV found under /kaggle/input. This Fitzpatrick mirror is "
    "images-only and has no ground-truth tone labels -> Arm A cannot run on "
    "it. Attach a mirror that ships fitzpatrick17k.csv (with a `fitzpatrick` "
    "column), e.g. try asosenge/fitzpatrick17k, then re-run.")
meta = pd.read_csv(csvs[0])
print("using CSV:", csvs[0].replace(INPUT, ""))
print("meta columns:", list(meta.columns))

cols = {c.lower(): c for c in meta.columns}
type_col = next((cols[c] for c in cols if "fitzpatrick" in c), None)
target_col = next((cols[c] for c in cols
                   if c in ("three_partition_label", "nine_partition_label",
                            "label", "condition", "diagnosis")), None)
key_col = next((cols[c] for c in cols
                if c in ("md5hash", "md5", "hash", "filename", "image",
                         "file", "path", "image_name", "url")),
               None)
assert type_col, f"No Fitzpatrick-type column in CSV. Columns: {list(meta.columns)}"
assert target_col, f"No condition/label column in CSV. Columns: {list(meta.columns)}"
assert key_col, f"No filename/md5 key column in CSV. Columns: {list(meta.columns)}"
print(f"type_col={type_col}  target_col={target_col}  key_col={key_col}")

def stem(x):
    x = str(x).split("/")[-1].split("\\")[-1]
    return os.path.splitext(x)[0]

# primary join: CSV key stem == image filename stem
path_by_stem = {stem(p): p for p in img_paths}
meta["_path"] = meta[key_col].map(lambda k: path_by_stem.get(stem(k)))
retr = meta["_path"].notna().mean()
print(f"primary join (by filename stem) retrieval: {retr*100:.1f}%")

# fallback join: some mirrors name files as "<label><n>_<k>.jpg" with no key
# match. If the primary join failed, match by condition label + row order so
# at least the tone/label pairing is preserved (reported as a caveat).
if retr < 0.05:
    print("!! primary join failed -- filenames do not match the CSV key column.")
    print("   Paste NB output to me; do NOT trust results until the join is "
          "confirmed. Attempting a label-ordered fallback...")
    lbl = meta[target_col].astype(str).str.lower().str.replace(r"[^a-z0-9]", "",
                                                               regex=True)
    meta = meta.assign(_lbl=lbl)
    def img_lbl(p):
        s = stem(p).lower()
        return "".join(ch for ch in s if ch.isalpha())
    img_df = pd.DataFrame({"_path2": img_paths})
    img_df["_lbl"] = img_df["_path2"].map(img_lbl)
    # naive per-label positional pairing
    meta["_ord"] = meta.groupby("_lbl").cumcount()
    img_df["_ord"] = img_df.groupby("_lbl").cumcount()
    meta = meta.merge(img_df, on=["_lbl", "_ord"], how="left")
    meta["_path"] = meta["_path"].fillna(meta["_path2"])
    retr = meta["_path"].notna().mean()
    print(f"fallback join retrieval: {retr*100:.1f}% (treat as approximate)")

df = meta[meta["_path"].notna()].copy()
assert len(df) > 200, (
    f"Only {len(df)} images matched labels -- too few to train. The mirror's "
    "filenames don't line up with its CSV. Send me the diagnostic output.")
print(f"metadata rows matched to an image file: {len(df)} / {len(meta)} "
      f"({100*len(df)/len(meta):.1f}% retrieval)")

# tone band from ground-truth type
df["fitz_num"] = pd.to_numeric(df[type_col], errors="coerce")
df = df[df["fitz_num"].between(1, 6)]
def band(t):
    return "Light" if t <= 2 else "Medium" if t <= 4 else "Deep"
df["tone_band"] = df["fitz_num"].map(band)

# target -> integer classes
df = df[df[target_col].notna()].copy()
classes = sorted(df[target_col].astype(str).unique())
cls2idx = {c: i for i, c in enumerate(classes)}
df["y"] = df[target_col].astype(str).map(cls2idx)
print("\ntarget classes:", cls2idx)
print("\ntone-band counts:\n", df["tone_band"].value_counts())
print("\nclass x tone-band:\n", pd.crosstab(df["y"], df["tone_band"]))

if CFG["max_images"]:
    df = df.groupby("y", group_keys=False).apply(
        lambda g: g.sample(min(len(g), CFG["max_images"] // len(classes)),
                            random_state=CFG["seed"]))
    print("subsampled to", len(df))


# %% [markdown]
# ## 2. Stratified split (jointly by class and tone band)
# %%
df["strata"] = df["y"].astype(str) + "_" + df["tone_band"]
vc = df["strata"].value_counts(); rare = vc[vc < 3].index
df.loc[df["strata"].isin(rare), "strata"] = "rare"
trv, te = train_test_split(df, test_size=0.2, random_state=CFG["seed"],
                           stratify=df["strata"])
tr, va = train_test_split(trv, test_size=0.125, random_state=CFG["seed"],
                          stratify=trv["strata"])
print(f"train {len(tr)} | val {len(va)} | test {len(te)}")


# %% [markdown]
# ## 3. Dataset + augmentation
# Colour/brightness jitter is kept minimal so it does not distort the tone
# signal the fairness audit depends on (§7.1).
# %%
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.05, contrast=0.05),   # deliberately small
    T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.ToTensor(), T.Normalize(MEAN, STD)])

class FitzDS(Dataset):
    def __init__(self, frame, tf):
        self.p = frame["_path"].values
        self.y = frame["y"].values
        self.tone = frame["tone_band"].values
        self.tf = tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB")
        return self.tf(img), int(self.y[i]), self.tone[i], self.p[i]

def collate(batch):
    xs, ys, tones, paths = zip(*batch)
    return torch.stack(xs), torch.tensor(ys), list(tones), list(paths)

dl = lambda frame, tf, shuf: DataLoader(
    FitzDS(frame, tf), batch_size=CFG["batch"], shuffle=shuf,
    num_workers=CFG["num_workers"], pin_memory=True, collate_fn=collate)
train_dl, val_dl, test_dl = dl(tr, train_tf, True), dl(va, eval_tf, False), dl(te, eval_tf, False)


# %% [markdown]
# ## 4. Model, class-weighted loss, train loop (mixed precision)
# %%
model = timm.create_model(CFG["model"], pretrained=True,
                          num_classes=len(classes)).to(DEVICE)
counts = tr["y"].value_counts().sort_index().values
cls_w = torch.tensor(counts.sum() / (len(counts) * counts),
                     dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=cls_w)
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                        weight_decay=CFG["weight_decay"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")

@torch.no_grad()
def evaluate(loader):
    model.eval(); ys, ps = [], []
    for x, y, _, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy()); ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")

best_f1, best_path = -1, os.path.join(WORK, "armA_fitz_resnet50_best.pt")
hist = []
for ep in range(CFG["epochs"]):
    model.train(); t0 = time.time(); running = 0
    for x, y, _, _ in train_dl:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        running += loss.item() * len(x)
    sched.step()
    va_acc, va_f1 = evaluate(val_dl)
    hist.append(dict(epoch=ep, train_loss=running/len(tr),
                     val_acc=va_acc, val_f1=va_f1))
    print(f"ep {ep:02d} | loss {running/len(tr):.4f} | "
          f"val_acc {va_acc:.4f} | val_f1 {va_f1:.4f} | {time.time()-t0:.0f}s")
    if va_f1 > best_f1:
        best_f1 = va_f1
        torch.save({"model": model.state_dict(), "cls2idx": cls2idx,
                    "cfg": CFG}, best_path)
        print("   * saved best")
pd.DataFrame(hist).to_csv(os.path.join(WORK, "armA_history.csv"), index=False)


# %% [markdown]
# ## 5. Test predictions WITH tone labels (the fairness-audit artifact)
# Saves per-test-image true/pred/probabilities + tone band. This CSV is the
# input to the disaggregated fairness audit (per-group accuracy, calibration,
# equalised-odds, bootstrapped CIs).
# %%
ckpt = torch.load(best_path, map_location=DEVICE)
model.load_state_dict(ckpt["model"]); model.eval()
rows = []
with torch.no_grad():
    for x, y, tones, paths in test_dl:
        x = x.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
            prob = torch.softmax(model(x), 1).cpu().numpy()
        pred = prob.argmax(1)
        for j in range(len(paths)):
            r = dict(path=paths[j], tone_band=tones[j],
                     y_true=int(y[j]), y_pred=int(pred[j]),
                     confidence=float(prob[j].max()))
            for c in range(len(classes)):
                r[f"prob_{c}"] = float(prob[j][c])
            rows.append(r)
pred_df = pd.DataFrame(rows)
pred_df.to_csv(os.path.join(WORK, "armA_test_predictions.csv"), index=False)

acc = accuracy_score(pred_df["y_true"], pred_df["y_pred"])
f1m = f1_score(pred_df["y_true"], pred_df["y_pred"], average="macro")
print(f"\nARM A TEST: acc={acc:.4f}  macro-F1={f1m:.4f}  (best val F1 {best_f1:.4f})")
print("\nPer-tone-band accuracy (headline fairness signal):")
for b in ["Light", "Medium", "Deep"]:
    g = pred_df[pred_df["tone_band"] == b]
    if len(g):
        print(f"  {b:7s} n={len(g):5d}  acc={accuracy_score(g.y_true, g.y_pred):.4f}  "
              f"macroF1={f1_score(g.y_true, g.y_pred, average='macro'):.4f}")
print("\nclasses:", cls2idx)
print(classification_report(pred_df["y_true"], pred_df["y_pred"]))

with open(os.path.join(WORK, "armA_summary.json"), "w") as fh:
    json.dump(dict(test_acc=acc, test_macro_f1=f1m, best_val_f1=best_f1,
                   classes=cls2idx, n_test=len(pred_df)), fh, indent=2)
print("\nSaved: armA_fitz_resnet50_best.pt, armA_test_predictions.csv, "
      "armA_history.csv, armA_summary.json")
print("NB2 (Arm A) complete.")
