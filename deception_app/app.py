"""
app.py
------
Flask backend for the thesis demo:
"Privacy-Preserving Multilingual Deception Detection System"

What this file does:
1. Defines the model architecture (attention pooling + classifier head) that
   matches EXACTLY what was used during federated training (FedProx, Flower),
   so that a checkpoint trained in the notebooks can be loaded here directly.
2. Loads the BanglishBERT encoder (csebuetnlp/banglishbert) + the trained
   classifier weights (the aggregated global model produced by the federated
   server) at server startup, ONCE, so /predict is fast.
3. Exposes:
     GET  /          -> serves the single-page demo (templates/index.html)
     POST /predict    -> runs inference, returns {prediction, confidence}

NOTE for supervisor demo:
This app expects a trained checkpoint at MODEL_CHECKPOINT_PATH (see below).
If that file is not found, the app still starts (so the UI can be shown),
but it falls back to the freshly-initialized classifier head and prints a
clear warning in the console. Predictions in that fallback mode are NOT
meaningful — they only exist so the interface can be demonstrated before
the real global-model checkpoint is copied in.
"""

import os
import re
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# HF checkpoint used for the BanglishBERT backbone (matches FILE2A in the thesis)
BASE_MODEL_NAME = "csebuetnlp/banglishbert"

# Path to the trained global model weights produced after FedProx aggregation.
# Put the .pt file the Flower server saved (e.g. the final round checkpoint,
# converted to a plain classifier-head + encoder state_dict) here.
MODEL_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model", "global_banglishbert.pt"
)

# Class labels — index 0 = Truthful, index 1 = Deceptive (matches training)
LABELS = {0: "Truthful", 1: "Deceptive"}

MAX_SEQ_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model architecture — must mirror the training notebooks exactly
# ---------------------------------------------------------------------------

class DeceptionClassifier(nn.Module):
    """
    Full model: pretrained multilingual encoder -> attention pooling ->
    classifier head.

    IMPORTANT: this class is written to be attribute-for-attribute and
    math-for-math IDENTICAL to `AttentionDeceptionClassifier` in the
    training notebook (Federated_banglishbert.ipynb, Cell 1). That notebook
    saves checkpoints with `torch.save(model.state_dict(), ...)`, and
    `load_state_dict` matches parameters by name -- so if this class' layer
    names or shapes drift from the notebook's, loading the trained
    checkpoint will fail with "Missing key(s)" / "Unexpected key(s)" errors.
    Do not rename `encoder`, `attention_pool`, or `classifier` below unless
    you make the same change in the notebook and retrain.
    """

    def __init__(self, base_model_name: str, dropout: float = 0.4):
        super().__init__()
        # IMPORTANT FIX (see thesis notes): some Hub checkpoints ship fp16
        # weights by default, which crashes against a float32 classifier
        # head. Force float32 explicitly.
        self.encoder = AutoModel.from_pretrained(
            base_model_name, torch_dtype=torch.float32
        ).float()

        hidden_size = self.encoder.config.hidden_size

        # Matches notebook: self.attention_pool = nn.Sequential(Linear, Tanh, Linear)
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        # Matches notebook: dropout on the first layer, dropout/2 on the second
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        scores = self.attention_pool(hidden)
        scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        pooled = (hidden * torch.softmax(scores, dim=1)).sum(dim=1)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Load tokenizer + model once at startup
# ---------------------------------------------------------------------------

print(f"[startup] Loading tokenizer: {BASE_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

print(f"[startup] Building model on device: {DEVICE}")
model = DeceptionClassifier(BASE_MODEL_NAME).to(DEVICE)

if os.path.exists(MODEL_CHECKPOINT_PATH):
    print(f"[startup] Loading trained federated global weights from: {MODEL_CHECKPOINT_PATH}")
    state_dict = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    MODEL_IS_TRAINED = True
else:
    print(
        "[startup][WARNING] No trained checkpoint found at "
        f"'{MODEL_CHECKPOINT_PATH}'.\n"
        "                    The app will still run so the UI can be demoed, "
        "but predictions are\n"
        "                    from an UNTRAINED classifier head and are not "
        "meaningful.\n"
        "                    Copy the trained global model .pt file into "
        "the model/ folder to fix this."
    )
    MODEL_IS_TRAINED = False

model.eval()


# ---------------------------------------------------------------------------
# Simple language heuristic — only used to label the "Auto" dropdown choice
# in the JSON response; it does not change how the text is tokenized, since
# BanglishBERT's tokenizer already handles English, Bangla and Banglish text.
# ---------------------------------------------------------------------------

BANGLA_UNICODE_RANGE = re.compile(r"[\u0980-\u09FF]")


def detect_language(text: str) -> str:
    if BANGLA_UNICODE_RANGE.search(text):
        return "Bangla"
    # crude Banglish heuristic: Latin script but contains common Banglish tokens
    banglish_markers = [" ami ", " tumi ", " kemon ", " bhalo ", " keno ", " ki "]
    padded = f" {text.lower()} "
    if any(marker in padded for marker in banglish_markers):
        return "Banglish"
    return "English"


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    """Serve the single-page demo UI."""
    return render_template("index.html", model_is_trained=MODEL_IS_TRAINED)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "text": "...", "language": "auto|english|bangla|banglish" }
    Returns JSON: { "prediction": "Truthful"|"Deceptive", "confidence": float,
                     "detected_language": "...", "model": "BanglishBERT (Federated)" }
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    language_choice = (data.get("language") or "auto").lower()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    # Resolve the language label to show in the result card
    if language_choice == "auto":
        resolved_language = detect_language(text)
    else:
        resolved_language = language_choice.capitalize()

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    return jsonify(
        {
            "prediction": LABELS[pred_idx],
            "confidence": round(confidence * 100, 2),
            "detected_language": resolved_language,
            "model": "BanglishBERT (Federated)",
            "model_is_trained": MODEL_IS_TRAINED,
        }
    )


if __name__ == "__main__":
    # debug=True is fine for a local thesis demo; turn off for any public deployment
    app.run(host="0.0.0.0", port=5000, debug=True)
