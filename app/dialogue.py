"""
dialogue.py
------------------------------------------------------------------
Rules-based dialogue manager: slot filling for the attributes the image
cannot supply (proposal section 7.7).

"A rules-based dialogue manager is retained as a fallback, and the system
remains fully functional without the language model, which sacrifices fluency
but nothing evaluable." -- this module IS that manager, and it is the default.

Responsibilities (strictly bounded):
  - ask for skin type, sensitivity, PIH concern, budget
  - parse free-text replies into structured slots
  - route medical requests to safety.py

It NEVER selects, reorders, filters, or introduces products. Recommendation
logic lives entirely in scripts/recommender.py.
------------------------------------------------------------------
"""
import re
from . import safety

SLOTS = ["skin_type", "sensitive", "pih_concern", "budget"]

QUESTIONS = {
    "skin_type": "How would you describe your skin most days -- oily, dry, "
                 "combination, or normal?",
    "sensitive": "Does your skin tend to sting, burn, or go red with new "
                 "products? (yes / no)",
    "pih_concern": "Are dark marks or spots left behind after spots heal a "
                   "concern for you? (yes / no)",
    "budget": "Is there a per-product budget you'd like me to stay under? "
              "(e.g. 40, or 'no limit')",
}

SKIN_TYPE_PATTERNS = {
    "oily":        [r"\boily\b", r"\bgreasy\b", r"\bshiny\b", r"\bshine\b"],
    "dry":         [r"\bdry\b", r"\bflaky\b", r"\btight\b", r"\bdehydrated\b"],
    "combination": [r"\bcombination\b", r"\bcombo\b", r"\bmixed\b",
                    r"oily.*t.?zone", r"t.?zone.*oily"],
    "normal":      [r"\bnormal\b", r"\bbalanced\b", r"\bfine\b"],
}
YES = re.compile(r"\b(yes|yeah|yep|yup|sure|often|sometimes|a bit|kind of|"
                 r"definitely|very|true|y)\b", re.I)
NO = re.compile(r"\b(no|nope|not really|never|rarely|none|nah|false|n)\b", re.I)


def parse_skin_type(text):
    t = text.lower()
    # check combination first: "oily t-zone" should not match plain "oily"
    for st in ("combination", "oily", "dry", "normal"):
        if any(re.search(p, t) for p in SKIN_TYPE_PATTERNS[st]):
            return st
    return None


def parse_yes_no(text):
    # negation wins: "no, not sensitive" must not read as yes
    if NO.search(text):
        return False
    if YES.search(text):
        return True
    return None


def parse_budget(text):
    t = text.lower()
    if re.search(r"\b(no limit|any|unlimited|doesn'?t matter|no budget|"
                 r"whatever)\b", t):
        return "none"
    m = re.search(r"(\d+(?:\.\d+)?)", t.replace(",", ""))
    return float(m.group(1)) if m else None


class DialogueManager:
    """Deterministic slot-filling conversation. No language model required."""

    def __init__(self):
        self.slots = {s: None for s in SLOTS}
        self.transcript = []

    # ------------------------------------------------------------------ state
    def next_slot(self):
        for s in SLOTS:
            if self.slots[s] is None:
                return s
        return None

    def complete(self):
        return self.next_slot() is None

    def ask(self):
        s = self.next_slot()
        return QUESTIONS[s] if s else None

    # ------------------------------------------------------------------ input
    def handle(self, message):
        """Process one user message. Returns a dict describing what happened.
        `refused` messages never advance the slot state."""
        self.transcript.append(("user", message))

        cat, refusal = safety.check(message)
        if cat:
            self.transcript.append(("system", refusal))
            return dict(kind="refusal", category=cat, response=refusal,
                        slot=None)

        slot = self.next_slot()
        if slot is None:
            resp = "I have everything I need -- generating your suggestions."
            self.transcript.append(("system", resp))
            return dict(kind="complete", category=None, response=resp,
                        slot=None)

        value = {"skin_type": parse_skin_type,
                 "sensitive": parse_yes_no,
                 "pih_concern": parse_yes_no,
                 "budget": parse_budget}[slot](message)

        if value is None:
            resp = ("Sorry, I didn't catch that. " + QUESTIONS[slot])
            self.transcript.append(("system", resp))
            return dict(kind="reprompt", category=None, response=resp,
                        slot=slot)

        self.slots[slot] = value
        nxt = self.ask()
        resp = nxt if nxt else ("Thanks -- that's everything I need.")
        self.transcript.append(("system", resp))
        return dict(kind="filled", category=None, response=resp, slot=slot,
                    value=value)

    # ------------------------------------------------------------------ output
    def to_recommender_kwargs(self):
        """Translate filled slots into recommender arguments. This is the ONLY
        thing the dialogue contributes to ranking."""
        b = self.slots["budget"]
        return dict(skin_type=self.slots["skin_type"],
                    sensitive=bool(self.slots["sensitive"]),
                    pih_concern=bool(self.slots["pih_concern"]),
                    budget=None if b in (None, "none") else float(b))
