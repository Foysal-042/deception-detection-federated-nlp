# Privacy-Preserving Deception Detection in Multilingual Computer-Mediated Communication via Federated Attention-Based Models

Thesis project (CUET, Dept. of CSE) exploring whether multilingual transformer
models can detect deception (fake news, phishing, job scams, product reviews,
political statements) while preserving data privacy through **federated
learning**, across **English, Bangla, and Banglish** text.

## Problem

Deception detection datasets are sensitive and domain-specific (e.g. phishing
emails, political claims). Centralizing this data raises privacy concerns.
This project asks: **how much accuracy do we lose by training a shared model
without centralizing the data?**

## Approach

- 5 domains (Fake News, Product Reviews, Phishing, Political Statements, Job
  Scams) modeled as 5 federated clients — each with its own task, zero data
  overlap.
- Multilingual text: 60% English, 30% Bangla, 10% Banglish.
- Same architecture across all backbones: pretrained encoder → attention
  pooling → classifier head.
- 5 backbone encoders compared: BanglishBERT, MuRIL, mBERT, DistilXLM-R,
  ALBERT Multilingual.
- Federated training via **Flower** with the **FedProx** strategy (10 rounds,
  domain-partitioned non-IID clients).
- Each backbone trained both ways: centralized baseline vs. federated, to
  measure accuracy retention.

## Results

**Table 1 — Centralized baseline**

| Backbone | Accuracy | Precision | Recall | Macro-F1 | AUC-ROC | MCC |
|---|---|---|---|---|---|---|
| BanglishBERT | 0.8285 | 0.8286 | 0.8285 | 0.8285 | 0.9154 | 0.6571 |
| mBERT | 0.8181 | 0.8181 | 0.8181 | 0.8181 | 0.9182 | 0.6363 |
| DistilXLM-R | 0.8151 | 0.8152 | 0.8151 | 0.8150 | 0.9157 | 0.6303 |
| MuRIL | 0.8135 | 0.8187 | 0.8136 | 0.8128 | 0.9124 | 0.6322 |
| ALBERT Multilingual | 0.7005 | 0.7070 | 0.7005 | 0.6981 | 0.7892 | 0.4074 |

**Table 2 — Federated (FedProx, 5 domain-partitioned clients)**

| Backbone | Accuracy | Precision | Recall | Macro-F1 | AUC-ROC | MCC |
|---|---|---|---|---|---|---|
| BanglishBERT | 0.7598 | 0.7703 | 0.7597 | 0.7574 | 0.8592 | 0.5300 |
| MuRIL | 0.7544 | 0.7715 | 0.7543 | 0.7504 | 0.8326 | 0.5256 |
| DistilXLM-R | 0.7436 | 0.7553 | 0.7436 | 0.7406 | 0.8356 | 0.4988 |
| mBERT | 0.7336 | 0.7517 | 0.7335 | 0.7287 | 0.8423 | 0.4849 |
| ALBERT Multilingual | 0.6608 | 0.7201 | 0.7081 | 0.6604 | 0.7839 | 0.4281 |

**Accuracy retention (federated F1 / centralized F1, same backbone)**

| Backbone | Retention |
|---|---|
| ALBERT Multilingual | 94.3% |
| MuRIL | 92.7% |
| DistilXLM-R | 91.2% |
| BanglishBERT | 91.7% |
| mBERT | 89.7% |

All five backbones retain **>89% accuracy under federation**, showing the
privacy-utility tradeoff is modest even under an extreme non-IID setup
(5 clients, each holding one entire domain).

## Repository structure

```
notebooks/
  centralized/    5 notebooks — non-federated baseline per backbone
  federated/      5 notebooks — Flower + FedProx per backbone
src/              reusable model + client code
prototype-app/    Flask demo app for inference (see its own README)
diagrams/         architecture diagrams (transformer, FL pipeline, overall pipeline)
results/          results_summary.csv (all runs, all metrics)
thesis/           full thesis report (PDF)
```

## Tech stack

PyTorch, HuggingFace Transformers, Flower (`flwr[simulation]`), Ray, Flask.
Trained on Kaggle (2× NVIDIA T4 GPU).

## Reproducing

Model checkpoints are not included in this repo (485MB, exceeds GitHub's
size limits). To reproduce: run the notebooks in `notebooks/` on Kaggle or
locally with a GPU; see `prototype-app/model/README.txt` for how to slot a
trained checkpoint into the demo app.

## Full thesis

See `thesis/` for the complete report, including literature review,
methodology, and full result analysis.
