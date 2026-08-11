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

from app.dialogue import DialogueManager          # noqa: E402
from app import safety                            # noqa: E402
from app.inference import (SeverityModel, ABSTAIN_MESSAGE,   # noqa: E402
                           GRADE_NAMES)
from recommender import SkincareRecommender       # noqa: E402

st.set_page_config(page_title="Skincare assistant (research demonstrator)",
                   page_icon="*", layout="wide")


@st.cache_resource
def get_model():
    return SeverityModel()


@st.cache_resource
def get_recommender():
    return SkincareRecommender()


# ----------------------------------------------------------------- state
if "dm" not in st.session_state:
    st.session_state.dm = DialogueManager()
    st.session_state.chat = []
    st.session_state.assessment = None
    st.session_state.done = False

dm = st.session_state.dm

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About this demonstrator")
    st.warning(safety.DISCLAIMER)
    st.caption(
        "Research artefact for an MSc dissertation on fairness in automated "
        "skin assessment. Recommendation logic is fully deterministic; no "
        "language model influences which products are shown."
    )
    st.subheader("Collected so far")
    for k, v in dm.slots.items():
        st.write(f"- **{k}**: {v if v is not None else '_not yet asked_'}")
    if st.button("Start over"):
        for key in ("dm", "chat", "assessment", "done"):
            st.session_state.pop(key, None)
        st.rerun()

st.title("Skincare assistant")
st.caption("Cosmetic recommendations from an acne-severity model plus a short "
           "conversation. Not a medical device.")

# ----------------------------------------------------------------- image
col_img, col_chat = st.columns([1, 2])

with col_img:
    st.subheader("1. Photo (optional)")
    up = st.file_uploader("Upload a clear, evenly lit photo",
                          type=["jpg", "jpeg", "png"])
    if up is not None and st.session_state.assessment is None:
        img = Image.open(up)
        st.image(img, use_container_width=True)
        with st.spinner("Assessing..."):
            st.session_state.assessment = get_model().predict(img)

    a = st.session_state.assessment
    if a:
        if a["abstain"]:
            st.info(ABSTAIN_MESSAGE)
            st.caption(f"Confidence {a['confidence']:.2f} is below the "
                       f"derived threshold {a['threshold']:.2f}.")
        else:
            st.success(f"Assessed severity: **{a['grade_name']}** "
                       f"(confidence {a['confidence']:.2f})")
            if a["referral"]:
                st.error("At this severity, please also speak to a GP or "
                         "dermatologist. The suggestions below are cosmetic "
                         "support, not a treatment plan.")
        with st.expander("Why this assessment? (auditability)"):
            st.write("Class probabilities:")
            for g, p in a["probabilities"].items():
                st.write(f"- {GRADE_NAMES.get(g, g)}: {p:.3f}")
            st.caption("Abstention threshold derived in "
                       "reports/08_abstention_threshold.md")

# ----------------------------------------------------------------- chat
with col_chat:
    st.subheader("2. A few questions")
    if not st.session_state.chat:
        opening = dm.ask()
        st.session_state.chat.append(("assistant", opening))

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    if not st.session_state.done:
        user_msg = st.chat_input("Type your answer...")
        if user_msg:
            st.session_state.chat.append(("user", user_msg))
            result = dm.handle(user_msg)
            st.session_state.chat.append(("assistant", result["response"]))
            if dm.complete():
                st.session_state.done = True
            st.rerun()

# ----------------------------------------------------------------- results
if st.session_state.done:
    st.divider()
    st.subheader("3. Suggested products")

    a = st.session_state.assessment
    rec = get_recommender()
    kwargs = dm.to_recommender_kwargs()

    if a is None or a["abstain"]:
        grade = 0
        st.info("No confident severity assessment, so these are based on your "
                "answers alone and assume a mild presentation.")
    else:
        grade = a["grade"]

    recs = rec.recommend(grade=grade, k=8, **kwargs)

    st.caption(f"Ranked deterministically for grade **{GRADE_NAMES[grade]}**, "
               f"skin type **{kwargs['skin_type']}**, "
               f"sensitive **{kwargs['sensitive']}**, "
               f"dark-mark concern **{kwargs['pih_concern']}**, "
               f"budget **{kwargs['budget'] or 'none'}**.")
    st.dataframe(
        recs[["product_name", "brand_name", "price_usd", "mean_rating",
              "n_reviews", "secondary_category"]],
        use_container_width=True, hide_index=True)

    with st.expander("How were these chosen? (full traceability)"):
        st.write(
            "1. The CNN produced a severity grade (or abstained).\n"
            "2. The **mapping table** (reports/05_mapping_table.md) converts "
            "that grade plus your declared attributes into preferred and "
            "de-prioritised ingredient classes.\n"
            "3. The **ranking function** scores every catalogue product on "
            "ingredient match, review sentiment, rating, skin-type fit, and "
            "an irritancy penalty if you reported sensitivity.\n\n"
            "No language model participates in any of these steps."
        )
        st.write("Reference ingredient classes endorsed for this grade:")
        st.code(", ".join(sorted(rec.acceptable_ingredient_classes(grade))))

    st.warning(safety.DISCLAIMER)
