"""
safety.py
------------------------------------------------------------------
Refusal-and-redirect behaviour for the conversational interface
(proposal sections 7.7 and 11).

A conversational interface invites requests a form does not -- principally
medical ones. This module specifies that behaviour IN ADVANCE and makes it
testable: it is deterministic, inspectable, and never delegated to a language
model's discretion.

Categories are ordered by severity; the first match wins, so a message that
mentions both a diagnosis request and a product question is refused.
------------------------------------------------------------------
"""
import re

# Ordered: most serious first.
REFUSAL_RULES = [
    ("malignancy", [
        r"\b(cancer|melanoma|carcinoma|malignan\w*|tumou?r|biopsy)\b",
        r"\bis (this|it) (a )?(mole|lesion|spot).*(dangerous|cancer|serious)",
        r"\bshould i (be )?worr(y|ied)\b.*\b(mole|spot|lesion|growth)\b",
    ], "I can't assess whether a lesion is cancerous, and I won't guess. "
       "Anything changing in size, shape, colour, or that bleeds or won't "
       "heal, should be looked at by a doctor or dermatologist promptly."),

    # NOTE: several patterns below were widened after the adversarial test in
    # scripts/09_adversarial_test.py caught them missing real phrasings
    # (plurals, reversed word order, non-"how do I" framings). The test set is
    # the regression guard for exactly this.
    ("emergency", [
        # bidirectional: "throat is swelling" and "swelling in my throat"
        r"\b(swell\w*|swollen)\b.{0,40}\b(throat|lip|lips|face|tongue)\b",
        r"\b(throat|lip|lips|face|tongue)\b.{0,40}\b(swell\w*|swollen)\b",
        r"\b(difficulty|trouble|hard)\b.{0,20}\bbreath\w*",
        r"\bcan'?t breathe\b",
        r"\banaphyla\w*",
        r"\bspreading\b.{0,40}\b(rash|redness|red)\b.{0,40}\bfever\b",
        r"\b(rash|redness)\b.{0,40}\b(fever|spreading)\b",
    ], "That could be a medical emergency. Please seek urgent medical care "
       "now -- call your local emergency number or go to A&E. I'm not able to "
       "help with this."),

    ("diagnosis", [
        r"\b(diagnos\w+)\b",
        r"\bwhat\b.{0,30}\b(disease|condition|infection|illness)\b",
        r"\bdo i have\b.*\b(rosacea|eczema|psoriasis|dermatitis|fungal|infection|hs|lupus)\b",
        r"\bis (this|it)\b.*\b(rosacea|eczema|psoriasis|dermatitis|infection|staph|herpes)\b",
    ], "I can't diagnose skin conditions -- I only assess visible acne "
       "severity to suggest cosmetic products. If you think something else is "
       "going on, a GP or dermatologist can examine it properly."),

    ("prescription", [
        r"\b(isotretinoin|accutane|roaccutane|tretinoin|adapalene|"
        r"antibiotics?|doxycycline|minocycline|clindamycin|spironolactone|"
        r"steroids?|hydrocortisone|prescriptions?|prescribe)\b",
        r"\bwhat (dose|dosage|mg)\b",
        r"\b(stop|start|come off|quit)\b.{0,25}\b(medication|meds|tablets?|pills?)\b",
        r"\bcan i (take|use)\b.*\b(pill|tablet|medication|drug)\b",
    ], "I can't advise on prescription treatments or dosages. That's a "
       "conversation for a doctor or dermatologist, who can weigh your history "
       "and monitor side effects. I can only suggest over-the-counter "
       "cosmetic products."),

    ("self_harm_procedure", [
        r"\b(pop|lance|cut open|drain|extract|squeeze)\b.{0,30}\b(cyst|nodule|boil|abscess|spot)\b",
        r"\bat home\b.{0,30}\b(surgery|excision|remove|removal)\b",
        # any self-removal framing, not just "how do I"
        r"\b(remove|removing|cut off|cutting off|burn off|freeze off)\b"
        r".{0,30}\b(mole|skin tag|wart|growth)\b",
        r"\b(mole|skin tag|wart|growth)\b.{0,30}\b(at home|myself|yourself)\b",
    ], "Please don't try that at home -- removing or draining lesions "
       "yourself risks scarring and infection. A clinician can do it safely."),
]

COMPILED = [(name, [re.compile(p, re.IGNORECASE) for p in pats], msg)
            for name, pats, msg in REFUSAL_RULES]

DISCLAIMER = ("This is a cosmetic product recommendation tool, not a medical "
              "device. It does not diagnose. For persistent, painful, or "
              "worsening skin problems, please see a qualified clinician.")


def check(message):
    """Return (category, response) if the message must be refused, else
    (None, None). Deterministic and order-sensitive by design."""
    for name, pats, msg in COMPILED:
        if any(p.search(message) for p in pats):
            return name, msg
    return None, None


def is_refusal(message):
    return check(message)[0] is not None
