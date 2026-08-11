"""
inference.py
------------------------------------------------------------------
Acne severity assessment for the demonstrator (proposal section 7.7).

This is the ONLY component that looks at the image, and it determines the
severity grade that the mapping table then acts on. It abstains below the
threshold derived in scripts/08_abstention_threshold.py rather than emitting
a low-confidence guess.
------------------------------------------------------------------
"""
import os
import json
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT_CANDIDATES = [
    os.path.join(ROOT, "data", "trained weights", "armB_acne_resnet50_best.pt"),
    os.path.join(ROOT, "data", "kaggle_out", "armB_acne_resnet50_best.pt"),
]
CFG_PATH = os.path.join(HERE, "abstention_config.json")

GRADE_NAMES = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Very severe"}
REFERRAL_GRADES = {2, 3}

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


def _find_checkpoint():
    for p in CKPT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "armB_acne_resnet50_best.pt not found. Expected in "
        "data/trained weights/ or data/kaggle_out/.")


def load_threshold():
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH) as fh:
            return json.load(fh)
    return dict(threshold=0.85, derivation="default (config missing)")


class SeverityModel:
    """Wraps the trained ResNet-50. Loaded once, reused per request."""

    def __init__(self, device=None):
        import timm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = _find_checkpoint()
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.classes = ck.get("classes", [0, 1, 2, 3])
        self.model = timm.create_model("resnet50", pretrained=False,
                                       num_classes=len(self.classes))
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.cfg = load_threshold()
        self.threshold = float(self.cfg.get("threshold", 0.85))
        self.checkpoint_path = path

    @torch.no_grad()
    def predict(self, image):
        """image: PIL.Image or path. Returns a dict including an explicit
        `abstain` flag -- the caller must honour it."""
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        image = image.convert("RGB")
        x = _TF(image).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.model(x), 1).cpu().numpy()[0]
        grade = int(prob.argmax())
        conf = float(prob.max())
        return dict(
            grade=grade,
            grade_name=GRADE_NAMES.get(grade, str(grade)),
            confidence=conf,
            probabilities={int(c): float(prob[i])
                           for i, c in enumerate(self.classes)},
            abstain=conf < self.threshold,
            threshold=self.threshold,
            referral=grade in REFERRAL_GRADES,
        )


ABSTAIN_MESSAGE = (
    "I can't assess this photo with enough confidence to advise on severity. "
    "Rather than guess, I'll leave the severity out. You can retake the photo "
    "in even, natural light with the affected area clearly in frame -- or I "
    "can suggest products based only on what you tell me about your skin."
)
