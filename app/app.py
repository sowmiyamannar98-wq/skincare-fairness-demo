"""
app.py -- Streamlit conversational demonstrator (proposal section 7.7)

Run:  streamlit run app/app.py

ARCHITECTURAL CONSTRAINT (section 7.7). Responsibilities are separated so that
the propagation analysis in RQ2 characterises the system it claims to measure:

  component        | responsibility                    | determines recs?
  -----------------|-----------------------------------|-----------------
  dialogue manager | phrasing, parsing replies to slots| NO
  CNN classifier   | acne severity from the image      | YES
  mapping table    | severity + attributes -> classes  | YES
  ranking function | product scoring and ordering      | YES

This build uses the RULES-BASED dialogue manager, so no language model is in
the loop at all -- the strongest possible form of that separation. Every
recommendation is traceable to a deterministic, inspectable path.

THIS FILE IS PRESENTATION ONLY. Quick-reply buttons submit ordinary text
through the same `DialogueManager.handle()` parser a typed reply uses, so the
interaction remains deterministic and the audit claim holds. Nothing here
selects, reorders, filters or introduces a product.

This is the Hugging Face Space copy, generated from the project repository.
Edit the originals in the repo and re-copy rather than editing here.
"""
import os
import sys
import streamlit as st
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# recommender.py resolves its parquet path relative to the working directory.
# Hugging Face launches the Space from the repo root, but pinning it here keeps
# the app runnable from anywhere without editing the audited module.
os.chdir(ROOT)

from app.dialogue import DialogueManager, SLOTS      # noqa: E402
from app import safety                               # noqa: E402
from app.inference import (SeverityModel, ABSTAIN_MESSAGE,   # noqa: E402
                           GRADE_NAMES)
from recommender import SkincareRecommender          # noqa: E402

st.set_page_config(page_title="Skincare assistant - research demonstrator",
                   page_icon=":sparkles:", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------- style
# Theme-aware: colours are driven by CSS variables that flip under Streamlit's
# dark theme, so the demonstrator is legible in either.
st.markdown("""
<style>
:root {
  --accent: #6C5CE7;
  --accent-2: #00B894;
  --card-bg: #ffffff;
  --card-br: #e6e8ef;
  --muted: #6b7280;
  --chip-bg: #f3f4f8;
  --shadow: 0 1px 3px rgba(16,24,40,.06), 0 1px 2px rgba(16,24,40,.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --card-bg: #171923; --card-br: #2b2f3d; --muted: #9aa1af;
    --chip-bg: #21242f;
    --shadow: 0 1px 3px rgba(0,0,0,.4);
  }
}
.block-container { padding-top: 2rem; max-width: 1150px; }

.hero {
  background: linear-gradient(120deg, var(--accent) 0%, #8E7CF7 55%, var(--accent-2) 100%);
  border-radius: 16px; padding: 22px 26px; margin-bottom: 6px; color: #fff;
}
.hero h1 { margin: 0 0 4px 0; font-size: 1.55rem; font-weight: 700; color:#fff; }
.hero p  { margin: 0; opacity: .92; font-size: .93rem; }
.hero .pill {
  display:inline-block; margin-top:10px; padding:3px 11px; border-radius:999px;
  background: rgba(255,255,255,.18); font-size:.74rem; letter-spacing:.3px;
}

.step { display:flex; align-items:center; gap:9px; padding:6px 0; font-size:.9rem; }
.step .dot {
  width:21px; height:21px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:.7rem; font-weight:700; flex:none;
  background: var(--chip-bg); color: var(--muted);
}
.step.done .dot { background: var(--accent-2); color:#fff; }
.step.active .dot { background: var(--accent); color:#fff; }
.step.done .lbl { color: var(--muted); text-decoration: line-through; }

.card {
  background: var(--card-bg); border:1px solid var(--card-br); border-radius:13px;
  padding:15px 17px; margin-bottom:11px; box-shadow: var(--shadow);
}
.card .top { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
.card .name { font-weight:650; font-size:1rem; line-height:1.3; }
.card .brand { color:var(--muted); font-size:.83rem; margin-top:2px; }
.card .price { font-weight:700; font-size:1.05rem; white-space:nowrap; }
.card .meta { color:var(--muted); font-size:.8rem; margin-top:7px; }
.chip {
  display:inline-block; padding:2px 9px; margin:5px 5px 0 0; border-radius:999px;
  background:var(--chip-bg); font-size:.72rem; color:var(--muted);
}
.chip.match { background: rgba(0,184,148,.14); color:#00875f; }
@media (prefers-color-scheme: dark) { .chip.match { color:#4ade9f; } }
.rank {
  display:inline-flex; width:20px; height:20px; border-radius:6px; margin-right:7px;
  background:var(--accent); color:#fff; font-size:.7rem; font-weight:700;
  align-items:center; justify-content:center; vertical-align:middle;
}
.banner {
  border-radius:11px; padding:12px 15px; margin:9px 0; font-size:.9rem;
  border-left:4px solid;
}
.banner.ok    { background:rgba(0,184,148,.10); border-color:var(--accent-2); }
.banner.warn  { background:rgba(245,158,11,.11); border-color:#f59e0b; }
.banner.stop  { background:rgba(239,68,68,.10);  border-color:#ef4444; }
.banner.info  { background:rgba(108,92,231,.10); border-color:var(--accent); }
.small { color:var(--muted); font-size:.82rem; }
div[data-testid="stSidebarUserContent"] { padding-top: .6rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_model():
    return SeverityModel()


@st.cache_resource(show_spinner=False)
def get_recommender():
    return SkincareRecommender()


PRETTY = {
    "salicylic_acid": "salicylic acid", "benzoyl_peroxide": "benzoyl peroxide",
    "azelaic_acid": "azelaic acid", "hyaluronic_acid": "hyaluronic acid",
    "vitamin_c": "vitamin C", "tranexamic_acid": "tranexamic acid",
    "alpha_arbutin": "alpha arbutin", "kojic_acid": "kojic acid",
    "licorice_root": "licorice root", "clay_charcoal": "clay / charcoal",
    "colloidal_oatmeal": "colloidal oatmeal", "tea_tree": "tea tree",
    "physical_spf": "mineral SPF", "essential_oil": "essential oil",
    "denatured_alcohol": "denatured alcohol",
}


def pretty(cls):
    return PRETTY.get(cls, cls.replace("_", " "))


SLOT_LABELS = {
    "skin_type": "Skin type",
    "sensitive": "Sensitivity",
    "pih_concern": "Dark marks",
    "budget": "Budget",
}

# Quick replies are a keyboard shortcut, not a separate code path: each button
# submits this literal string through DialogueManager.handle(), exactly as if
# the user had typed it.
QUICK = {
    "skin_type": ["Oily", "Dry", "Combination", "Normal"],
    "sensitive": ["Yes", "No"],
    "pih_concern": ["Yes", "No"],
    "budget": ["Under 25", "Under 50", "No limit"],
}

# ----------------------------------------------------------------- state
if "dm" not in st.session_state:
    st.session_state.dm = DialogueManager()
    st.session_state.chat = []
    st.session_state.assessment = None
    st.session_state.done = False
    st.session_state.pending = None
    st.session_state.model_error = None

dm = st.session_state.dm


def submit(msg):
    """Route one user message through the deterministic dialogue manager."""
    st.session_state.chat.append(("user", msg))
    result = dm.handle(msg)
    st.session_state.chat.append(("assistant", result["response"], result["kind"]))
    if dm.complete():
        st.session_state.done = True


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("#### Your photo")
    up = st.file_uploader("Clear, evenly lit, affected area in frame",
                          type=["jpg", "jpeg", "png"],
                          label_visibility="collapsed")
    if up is not None and st.session_state.assessment is None:
        img = Image.open(up)
        st.image(img, use_container_width=True)
        with st.spinner("Assessing severity..."):
            try:
                st.session_state.assessment = get_model().predict(img)
            except Exception as exc:                     # noqa: BLE001
                st.session_state.model_error = str(exc)
    elif up is not None:
        st.image(Image.open(up), use_container_width=True)

    if st.session_state.model_error:
        st.markdown(f"<div class='banner warn'>Severity model unavailable, so "
                    f"the assessment step is skipped and suggestions will rest "
                    f"on your answers alone.<br><span class='small'>"
                    f"{st.session_state.model_error[:180]}</span></div>",
                    unsafe_allow_html=True)

    st.markdown("#### Progress")
    nxt = dm.next_slot()
    photo_done = st.session_state.assessment is not None
    st.markdown(
        f"<div class='step {'done' if photo_done else 'active'}'>"
        f"<span class='dot'>{'✓' if photo_done else '1'}</span>"
        f"<span class='lbl'>Photo{' (optional)' if not photo_done else ''}</span></div>",
        unsafe_allow_html=True)
    for i, s in enumerate(SLOTS, start=2):
        filled = dm.slots[s] is not None
        cls = "done" if filled else ("active" if s == nxt else "")
        val = dm.slots[s]
        shown = {True: "yes", False: "no"}.get(val, val)
        label = SLOT_LABELS[s] + (f" - {shown}" if filled else "")
        st.markdown(f"<div class='step {cls}'><span class='dot'>"
                    f"{'✓' if filled else i}</span>"
                    f"<span class='lbl'>{label}</span></div>",
                    unsafe_allow_html=True)

    st.divider()
    if st.button("Start over", use_container_width=True):
        for key in ("dm", "chat", "assessment", "done", "pending",
                    "model_error"):
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown(f"<div class='banner warn'>{safety.DISCLAIMER}</div>",
                unsafe_allow_html=True)
    st.caption("Research artefact for an MSc dissertation on fairness in "
               "automated skin assessment. Recommendation logic is fully "
               "deterministic; no language model influences what is shown.")

# ----------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>Skincare assistant</h1>
  <p>Cosmetic suggestions from an acne-severity model plus a short conversation.</p>
  <span class="pill">DETERMINISTIC PIPELINE &nbsp;&middot;&nbsp; NO LANGUAGE MODEL &nbsp;&middot;&nbsp; NOT A MEDICAL DEVICE</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- assessment
a = st.session_state.assessment
if a:
    if a["abstain"]:
        st.markdown(f"<div class='banner info'><b>Severity not reported.</b><br>"
                    f"{ABSTAIN_MESSAGE}<br><span class='small'>Confidence "
                    f"{a['confidence']:.2f} is below the derived threshold "
                    f"{a['threshold']:.2f}.</span></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='banner ok'><b>Assessed severity: "
                    f"{a['grade_name']}</b> &nbsp;<span class='small'>"
                    f"confidence {a['confidence']:.2f}</span></div>",
                    unsafe_allow_html=True)
        if a["referral"]:
            st.markdown("<div class='banner stop'><b>Please also speak to a GP "
                        "or dermatologist.</b> At this severity the suggestions "
                        "below are cosmetic support, not a treatment plan.</div>",
                        unsafe_allow_html=True)

    with st.expander("Why this assessment?"):
        st.caption("Class probabilities from the ResNet-50 classifier:")
        for g, p in sorted(a["probabilities"].items()):
            st.progress(p, text=f"{GRADE_NAMES.get(g, g)} - {p:.3f}")
        st.caption("The model abstains below a threshold derived in "
                   "reports/08_abstention_threshold.md rather than emitting a "
                   "low-confidence guess.")

# ----------------------------------------------------------------- chat
if not st.session_state.chat:
    st.session_state.chat.append(("assistant", dm.ask(), "filled"))

for turn in st.session_state.chat:
    role, msg = turn[0], turn[1]
    kind = turn[2] if len(turn) > 2 else None
    with st.chat_message(role, avatar="🧑" if role == "user" else "🧴"):
        if kind == "refusal":
            st.markdown(f"<div class='banner stop'>{msg}</div>",
                        unsafe_allow_html=True)
        else:
            st.write(msg)

if st.session_state.pending is not None:
    msg = st.session_state.pending
    st.session_state.pending = None
    submit(msg)
    st.rerun()

if not st.session_state.done:
    slot = dm.next_slot()
    if slot in QUICK:
        cols = st.columns(len(QUICK[slot]))
        for c, label in zip(cols, QUICK[slot]):
            if c.button(label, use_container_width=True, key=f"q_{slot}_{label}"):
                st.session_state.pending = label
                st.rerun()

    if user_msg := st.chat_input("Type your answer, or ask me something..."):
        st.session_state.pending = user_msg
        st.rerun()

# ----------------------------------------------------------------- results
if st.session_state.done:
    st.divider()
    st.subheader("Suggested products")

    a = st.session_state.assessment
    rec = get_recommender()
    kwargs = dm.to_recommender_kwargs()

    if a is None or a["abstain"]:
        grade = 0
        st.markdown("<div class='banner info'>No confident severity "
                    "assessment, so these rest on your answers alone and "
                    "assume a mild presentation.</div>",
                    unsafe_allow_html=True)
    else:
        grade = a["grade"]

    recs = rec.recommend(grade=grade, k=8, return_scores=True, **kwargs)
    endorsed = rec.acceptable_ingredient_classes(grade)
    flag_cols = [c for c in recs.columns if c in set(rec.classes)]

    bits = [f"grade <b>{GRADE_NAMES[grade]}</b>",
            f"skin <b>{kwargs['skin_type'] or 'unspecified'}</b>"]
    if kwargs["sensitive"]:
        bits.append("<b>sensitive</b>")
    if kwargs["pih_concern"]:
        bits.append("<b>dark-mark concern</b>")
    if kwargs["budget"]:
        bits.append(f"under <b>${kwargs['budget']:.0f}</b>")
    st.markdown(f"<div class='small'>Ranked deterministically for "
                f"{' &middot; '.join(bits)}.</div>", unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2)
    for i, (_, row) in enumerate(recs.iterrows()):
        matched = [c for c in flag_cols if row.get(c, 0) == 1]
        hits = [c for c in matched if c in endorsed][:3]
        others = [c for c in matched if c not in endorsed][:2]
        price = row["price_usd"]
        chips = "".join(f"<span class='chip match'>{pretty(c)}</span>"
                        for c in hits)
        chips += "".join(f"<span class='chip'>{pretty(c)}</span>"
                         for c in others)
        rating = row["mean_rating"]
        nrev = row["n_reviews"]
        rating_txt = ("&#9733; %.2f" % rating) if rating == rating else "no rating"
        if not chips:
            chips = "<span class='chip'>no flagged actives</span>"
        card = (
            f"<div class='card'><div class='top'><div>"
            f"<div class='name'><span class='rank'>{i+1}</span>"
            f"{row['product_name']}</div>"
            f"<div class='brand'>{row['brand_name']}</div></div>"
            f"<div class='price'>${price:,.0f}</div></div>"
            f"<div class='meta'>{rating_txt}"
            f" &middot; {int(nrev):,} reviews"
            f" &middot; {row['secondary_category']}</div>"
            f"<div>{chips}</div>"
            f"</div>")
        (left if i % 2 == 0 else right).markdown(card, unsafe_allow_html=True)

    with st.expander("How were these chosen? (full traceability)"):
        st.markdown(
            "1. The **CNN** produced a severity grade from the photo, or "
            "abstained.\n"
            "2. The **mapping table** converts that grade plus your declared "
            "attributes into preferred and de-prioritised ingredient classes "
            "(`reports/05_mapping_table.md`).\n"
            "3. The **ranking function** scores every eligible product on "
            "ingredient match (1.00), review sentiment (0.35), star rating "
            "(0.25), skin-type fit (0.20) and an irritancy penalty (-0.30) "
            "applied only if you reported sensitivity.\n\n"
            "No language model participates in any of these steps, which is "
            "what makes the propagation analysis in the dissertation a "
            "measurement of this exact path.")
        st.caption("Ingredient classes endorsed for this grade:")
        st.code(", ".join(pretty(c) for c in sorted(endorsed)))

    st.markdown(f"<div class='banner warn'>{safety.DISCLAIMER}</div>",
                unsafe_allow_html=True)
