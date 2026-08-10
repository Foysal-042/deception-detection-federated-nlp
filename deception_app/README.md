# Privacy-Preserving Multilingual Deception Detection System — Demo Interface

Single-page Flask demo built for a thesis supervisor presentation, showing how
the federated BanglishBERT model (trained with Flower + FedProx) can be
deployed for inference.

## Project structure

```
deception_app/
├── app.py                 # Flask backend + model architecture + /predict route
├── requirements.txt
├── model/
│   ├── global_banglishbert.pt   # <- put your trained checkpoint here (not included)
│   └── README.txt
├── templates/
│   └── index.html          # single page: overview + analyze + result sections
└── static/
    ├── style.css            # white background, blue accent, rounded cards
    └── script.js            # calls /predict and renders the result card
```

## Setup

```bash
cd deception_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Add your trained model

Copy your trained global model weights (the aggregated model after the final
FedProx round) into:

```
model/global_banglishbert.pt
```

It should be a `state_dict` saved from the exact `DeceptionClassifier` class
in `app.py` (encoder + attention pooling + classifier head). See
`model/README.txt` for details.

If this file is not present, the app still runs so you can demo the
interface, but it will print a startup warning and predictions will not be
meaningful (untrained head).

## Run

```bash
python app.py
```

Then open **http://localhost:5000** in a browser.

## Notes

- The "Language" dropdown is for display/labeling only — BanglishBERT's
  tokenizer natively handles English, Bangla, and Banglish text, so no
  separate preprocessing branch is required per language.
- Section 1 (Federated Learning Overview) is a static visualization built
  with plain HTML/CSS — no charting library, no animation, matching the
  academic/demo tone requested.
