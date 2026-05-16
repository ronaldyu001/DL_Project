"""Generate PDF figures for the report from the results CSVs."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "../results/our_results"
FIG_DIR = "figures"

ci_df = pd.read_csv(f"{RESULTS_DIR}/all_results_with_ci.csv")
all_df = pd.read_csv(f"{RESULTS_DIR}/all_results.csv")


# Figure 1: Forest plot of F1 with 95% bootstrap CIs
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, ds in zip(axes, ["CreditCard", "BankSim"]):
    df = ci_df[ci_df["dataset"] == ds].sort_values("f1").reset_index(drop=True)
    y_pos = np.arange(len(df))
    err_lo = (df["f1"] - df["f1_lo"]).clip(lower=0)
    err_hi = (df["f1_hi"] - df["f1"]).clip(lower=0)
    ax.errorbar(df["f1"], y_pos, xerr=[err_lo, err_hi],
                fmt="o", color="steelblue", capsize=4, markersize=6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("F1 Score (point estimate, 95% CI)")
    ax.set_title(ds)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 1)

plt.suptitle("Bootstrap 95% Confidence Intervals: F1 Score", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/forest_f1.pdf", bbox_inches="tight")
plt.close()
print("Saved forest_f1.pdf")


# Figure 2: Side-by-side F1 and AUC-PR comparison across datasets
cc = all_df[all_df["dataset"] == "CreditCard"].set_index("model_clean")
bs = all_df[all_df["dataset"] == "BankSim"].set_index("model_clean")
shared = list(bs.index)
cc = cc.loc[shared]

x = np.arange(len(shared))
w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].bar(x - w/2, cc["f1"], w, label="CreditCard", color="steelblue")
axes[0].bar(x + w/2, bs["f1"], w, label="BankSim", color="darkorange")
axes[0].set_xticks(x)
axes[0].set_xticklabels(shared, rotation=30, ha="right")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("F1 Score")
axes[0].set_title("F1: CreditCard vs BankSim")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)
axes[0].axvline(x=3.5, color="gray", linestyle="--", linewidth=0.8)

axes[1].bar(x - w/2, cc["auc_pr"], w, label="CreditCard", color="steelblue")
axes[1].bar(x + w/2, bs["auc_pr"], w, label="BankSim", color="darkorange")
axes[1].set_xticks(x)
axes[1].set_xticklabels(shared, rotation=30, ha="right")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("AUC-PR")
axes[1].set_title("AUC-PR: CreditCard vs BankSim")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)
axes[1].axvline(x=3.5, color="gray", linestyle="--", linewidth=0.8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cross_dataset_comparison.pdf", bbox_inches="tight")
plt.close()
print("Saved cross_dataset_comparison.pdf")


# Figure 3: Pipeline diagram (simple block diagram)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

def box(x, y, w, h, text, color="lightsteelblue"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

# Input
box(0.2, 2.5, 1.6, 1, "Training Data\n(transactions)", "lightyellow")

# Base models
box(2.6, 4.4, 1.8, 0.9, "FFN", "lightsteelblue")
box(2.6, 3.3, 1.8, 0.9, "XGBoost", "lightsteelblue")
box(2.6, 2.2, 1.8, 0.9, "Autoencoder", "lightcoral")
box(2.6, 1.1, 1.8, 0.9, "Isolation Forest", "lightcoral")

# Stack
box(5.2, 2.5, 1.8, 1, "4-feature\nmeta-stack", "lightgray")

# Meta-learner
box(7.6, 2.5, 2.2, 1, "Meta-learner\n(6 candidates)", "palegreen")

# Arrows
for y in [4.85, 3.75, 2.65, 1.55]:
    arrow(1.85, 3.0, 2.55, y)
for y in [4.85, 3.75, 2.65, 1.55]:
    arrow(4.45, y, 5.15, 3.0)
arrow(7.05, 3.0, 7.55, 3.0)

# Labels
ax.text(3.5, 5.6, "Supervised", ha="center", fontsize=9, style="italic", color="darkblue")
ax.text(3.5, 0.6, "Unsupervised (normal only)", ha="center", fontsize=9, style="italic", color="darkred")
ax.text(8.7, 1.9, "Final fraud\nprobability", ha="center", fontsize=9, style="italic")

plt.savefig(f"{FIG_DIR}/pipeline_diagram.pdf", bbox_inches="tight")
plt.close()
print("Saved pipeline_diagram.pdf")
