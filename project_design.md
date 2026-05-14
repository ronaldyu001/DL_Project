# Credit Card Fraud Detection — Project Design


---

## Core Idea: Heterogeneous Stacking Ensemble

Most fraud detection work uses either supervised classifiers or unsupervised anomaly detectors. Our approach combines both paradigms under a learned meta-layer:

```
X_train (normal only) ──► Autoencoder        ──► reconstruction error ──┐
X_train (normal only) ──► Isolation Forest   ──► anomaly score ─────────┤
                                                                          ├──► [Meta-Learner] ──► Final Score
X_train (all, K-fold) ──► FFN                ──► OOF fraud prob ─────────┤
X_train (all, K-fold) ──► XGBoost            ──► OOF fraud prob ──────────┘
```

**Why this combination is principled:**

| Model | Type | What it captures |
|---|---|---|
| FFN | Supervised | Explicit fraud patterns from labeled examples |
| XGBoost | Supervised | Nonlinear feature interactions, strong tabular baseline |
| Autoencoder | Unsupervised | What "normal" looks like — fraud = high reconstruction error |
| Isolation Forest | Unsupervised | Isolation-based anomaly — different geometric intuition than AE |

The FFN and XGBoost know what fraud looks like. The AE and IF know what normal looks like. They fail in different situations. The meta-learner learns when to trust each.

---

## Why This Is Novel

| Prior Work | Approach |
|---|---|
| Pumsirirat 2018 | Autoencoder + RBM — both unsupervised, fixed combination |
| Alarfaj 2022 | Many models tested individually, no ensemble |
| **This project** | Supervised (FFN, XGBoost) + Unsupervised (AE, IF) + **learned meta-combiner with proper stacking** |

The key contributions:
1. Combining supervised and unsupervised paradigms (not just multiple supervised models)
2. Proper stacking via out-of-fold predictions — avoids meta-learner leakage
3. Systematic comparison of six meta-learning algorithms
4. Generalizability validated on a second dataset (BankSim)

---

## Proper Stacking: Why OOF Matters

A naive ensemble trains base models on X_train, then uses their X_train predictions as meta-features. This leaks: the base models have already seen those samples, so their predictions are overfit — the meta-learner trains on artificially clean signal that won't generalize.

**Our approach — out-of-fold (OOF) stacking:**

1. Split X_train into K=5 folds (stratified)
2. For each fold: train FFN/XGBoost on K-1 folds, predict on the held-out fold
3. After all folds: every sample in X_train has one OOF prediction (never seen during its prediction)
4. Train meta-learner on these OOF predictions with y_train labels

This gives the meta-learner **~344 fraud training samples** (70% of 492) instead of ~74 (15% val set), and the predictions are out-of-sample so there is no leakage.

Unsupervised models (AE, IF) have no label leakage by design — they are trained only on normal transactions and simply predict on all splits.

---

## Meta-Learner Comparison

Six algorithms are tested as the meta-learner, each trained on the same 4-feature OOF stack `[FFN_prob, XGB_prob, AE_score, IF_score]`:

| Meta-Learner | Rationale |
|---|---|
| Logistic Regression | Linear baseline; interpretable coefficients show which base model the meta-learner trusts |
| Random Forest | Nonlinear; robust to outliers in base model predictions |
| Gradient Boosting | Strong on tabular; sequentially corrects errors |
| SVM (RBF) | Decision boundary approach; good when classes are separable in meta-feature space |
| MLP | Small neural meta-learner; can learn complex interactions between base model outputs |
| AdaBoost | Adaptive boosting; weights hard examples more heavily |

All thresholds are tuned on the validation set; all metrics are reported on the held-out test set.

---

## Evaluation Strategy

**Primary metric:** F1-score (harmonic mean of precision and recall — appropriate for imbalanced data)

**Full metric suite:**
- Precision, Recall — understand the FP/FN tradeoff
- AUC-PR (area under precision-recall curve) — threshold-independent; better than AUC-ROC for imbalanced data
- Confusion matrices — visualize TP/FP/FN/TN counts

**Threshold tuning — two strategies:**
1. **F1-optimal threshold:** sweep thresholds on val set, pick the one maximizing F1
2. **Cost-sensitive threshold:** minimize `10 × FN + 1 × FP` — missing a fraud is 10× more costly than a false alarm

Both are applied to all models, tuned on val, evaluated on test.

---

## Datasets

### Primary: Kaggle Credit Card Fraud
- 284,315 normal / 492 fraud (0.17% rate)
- Features V1–V28: PCA-transformed (anonymous), plus Time and Amount
- Hardest imbalance; standard benchmark

### Secondary: BankSim
- ~587,443 normal / ~7,200 fraud (~1.2% rate)
- Raw interpretable features: merchant category, transaction amount, customer age/gender
- Synthetic but realistic; completely different feature space from creditcard
- Generalizability: if ensemble gains hold here, they are not dataset-specific

**Split (both datasets):** 70% train / 15% val / 15% test, stratified by class label.

### BankSimNET (not used)
BankSimNET adds graph/network structure (customer-merchant edges) to BankSim, enabling Graph Neural Network approaches. Out of scope for this project but noted as future work — GNNs on BankSimNET could capture relational fraud patterns that our transaction-independent pipeline misses.

---

## Separate Baselines Notebook

A separate `baselines.ipynb` runs off-the-shelf algorithms directly on the raw features (no ensemble), establishing comparison points:

- Logistic Regression (floor)
- Random Forest
- XGBoost directly on raw features ← most important; if this beats our ensemble, the architecture story breaks
- SMOTE + any of the above (tests whether oversampling alone closes the gap)

If the ensemble pipeline beats the raw-feature XGBoost baseline, the added complexity is justified. If it doesn't, that is itself an important finding.

---

## Known Limitations

- **BankSim is synthetic** — real-world fraud patterns may differ
- **AE/IF are calibrated via sigmoid of standardized scores** — not probabilistically calibrated in the Platt/isotonic sense; future work could use proper calibration
- **No statistical significance testing** — with ~74 fraud cases in the test set, differences between models may not be significant; bootstrap confidence intervals on F1 would strengthen claims
- **FFN architecture not hyperparameter-tuned** — fixed 128→64→32 architecture; NAS or grid search could improve it
- **BankSimNET excluded** — graph-based approaches left as future work
