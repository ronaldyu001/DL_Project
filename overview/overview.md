# Credit Card Fraud Detection — Project Overview

## The Problem

The Kaggle dataset has ~284,000 normal transactions and only ~492 fraud — **0.17% fraud rate**. This extreme class imbalance is the core challenge. A model that always predicts "not fraud" achieves 99.8% accuracy but is completely useless.

---

## Architecture

### 1. Feedforward Neural Network (FFN) — Supervised

- Trained on the full dataset (fraud + normal transactions)
- Learns: "given these transaction features, what does a fraud look like?"
- Uses weighted loss so the model penalizes missing fraud more than missing normal
- Output: a probability `P(fraud)` between 0 and 1

**Strength:** catches fraud patterns it was explicitly trained on

**Weakness:** if a new fraud pattern emerges that looks different from training data, it will miss it

---

### 2. Autoencoder — Unsupervised Anomaly Detection

- Trained **only on normal transactions**
- Learns to compress and reconstruct what a normal transaction looks like
- At inference, run any transaction through it and measure **reconstruction error**
- High error = transaction doesn't look "normal" = likely anomaly
- Output: reconstruction error converted to a probability

**Strength:** catches novel or previously unseen fraud — anything that looks anomalous

**Weakness:** can flag unusual-but-legitimate transactions (e.g., a very large but valid purchase)

---

### 3. Learned Meta-Ensemble — The Novelty

- Takes both outputs (FFN probability + Autoencoder anomaly score) as inputs
- A small **logistic regression** trained on a held-out validation set
- Learns: "when should I trust the FFN more? When should I trust the Autoencoder more?"
- Output: final fraud probability

**Why this beats a fixed weighted average:** the FFN and Autoencoder fail in different situations. A fixed 50/50 (or any fixed) average cannot adapt to that. The meta-learner figures out the optimal combination from data.

---

## Pipeline

```
Raw Transaction
      |
   [FFN] ──────────────────────► P(fraud) ──────┐
                                                  ├──► [Meta Logistic Regression] ──► Final Score ──► Fraud / Not Fraud
   [Autoencoder] ──► Recon Error ──► Prob ────────┘
```

---

## Why This is Novel

| Paper                  | What They Did                                                                  |
| ---------------------- | ------------------------------------------------------------------------------ |
| Pumsirirat 2018        | Autoencoder + RBM — both unsupervised, fixed combination                      |
| Alarfaj 2022           | Tested many models individually, no ensemble                                   |
| **This project** | Supervised (FFN) + Unsupervised (Autoencoder) +**learned meta-combiner** |

The key insight: these two models are **complementary by design**. The FFN knows what fraud looks like; the Autoencoder knows what normal looks like. Neither is sufficient alone. The meta-learner arbitrates between them based on what the data shows.

---

## Evaluation

Three experiments, compared head-to-head:

1. FFN alone
2. Autoencoder alone
3. Meta-ensemble

### Metrics

-**F1-score** — primary metric given class imbalance
-**Precision / Recall** — understand the false positive / false negative tradeoff
-**AUC-PR** (area under precision-recall curve) — better than AUC-ROC for imbalanced data
-**Cost-sensitive threshold tuning** — optimize the decision threshold using a realistic cost ratio (missing a fraud costs more than a false alarm)

### Expected Story

The ensemble outperforms either individual model, and the learned combiner outperforms a naive fixed weighted average.

---

## Dataset

**Primary:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,315 normal transactions, 492 fraudulent
- Features V1–V28 are PCA-transformed (anonymous), plus `Time` and `Amount`

**Secondary (if time permits):** BankSim Synthetic Payments Dataset

- 594,643 records (587,443 normal, 7,200 fraudulent)
- Raw interpretable features — run same pipeline to validate ensemble gains hold across datasets

---

## Team

- Ronald Yu
- Joseph Tesoriero
- Tejal Jadhav

**Deadline:** May 9, 2026
