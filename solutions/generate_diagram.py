"""
Generates solutions/architecture_diagram.png — VaultTech AWS architecture.

Run with:
    uv run python solutions/generate_diagram.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUTPUT = Path(__file__).resolve().parent / "architecture_diagram.png"

# ── colour palette ────────────────────────────────────────────────────────────
C_AWS       = "#FF9900"   # AWS orange
C_AWS_DARK  = "#C47400"
C_BLUE      = "#1A73E8"   # user / browser
C_TEAL      = "#00897B"   # ECS / Fargate
C_PURPLE    = "#6A1B9A"   # SageMaker
C_GREEN     = "#2E7D32"   # S3
C_GRAY      = "#455A64"   # ECR
C_GOLD      = "#F57F17"   # Model Registry
C_BG        = "#F8F9FA"
C_BORDER    = "#CFD8DC"


def rounded_box(ax, x, y, w, h, color, label, sublabel=None, fontsize=10):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.8, edgecolor=color, facecolor=color + "22",
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + (0.08 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.13, sublabel,
                ha="center", va="center", fontsize=7.5, color="#546E7A", zorder=4)


def arrow(ax, x1, y1, x2, y2, label="", color="#546E7A", style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=1.5,
                        connectionstyle="arc3,rad=0.0"),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.05, label, ha="center", va="bottom",
                fontsize=7.5, color=color, style="italic", zorder=5)


def curved_arrow(ax, x1, y1, x2, y2, label="", color="#546E7A", rad=0.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=2,
    )
    if label:
        mx = (x1 + x2) / 2 + rad * 0.4
        my = (y1 + y2) / 2 + abs(rad) * 0.3
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=7.5, color=color, style="italic", zorder=5)


# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# title
ax.text(7, 7.6, "VaultTech — AWS Architecture", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#263238")
ax.text(7, 7.25, "Forging Line Bath Time Predictor  |  eu-west-1 (Ireland)",
        ha="center", va="center", fontsize=9, color="#607D8B")

# ── AWS region boundary ───────────────────────────────────────────────────────
region_box = FancyBboxPatch(
    (2.8, 0.4), 10.8, 6.5,
    boxstyle="round,pad=0.05",
    linewidth=2, edgecolor=C_AWS, facecolor="#FFF8E1", alpha=0.35, zorder=1,
)
ax.add_patch(region_box)
ax.text(3.25, 6.75, "AWS  eu-west-1", ha="left", va="center",
        fontsize=9, color=C_AWS_DARK, fontweight="bold", zorder=2)

# ── nodes ─────────────────────────────────────────────────────────────────────

# 1. User / Browser
rounded_box(ax, 1.3, 4.5, 1.8, 0.8, C_BLUE,
            "User Browser", "http://3.253.10.36:8501")

# 2. ECR
rounded_box(ax, 4.5, 6.2, 2.4, 0.75, C_GRAY,
            "Amazon ECR",
            "999390550986.dkr.ecr.eu-west-1\n.amazonaws.com/vaultech-app")

# 3. ECS / Fargate  (centre of diagram)
rounded_box(ax, 5.5, 4.5, 3.0, 1.1, C_TEAL,
            "ECS / Fargate", "vaultech-app-service\nStreamlit on port 8501\n0.5 vCPU · 1 GB RAM", fontsize=9)

# 4. SageMaker Endpoint
rounded_box(ax, 10.2, 4.5, 3.0, 1.1, C_PURPLE,
            "SageMaker Endpoint",
            "vaultech-bath-predictor\nXGBoost 3.0-5 container\nml.t2.medium · InService", fontsize=9)

# 5. S3 – model artifact
rounded_box(ax, 10.2, 2.3, 3.0, 0.85, C_GREEN,
            "Amazon S3",
            "vaultech-models-999390550986\nmodels/xgboost-bath-predictor/model.tar.gz")

# 6. Model Registry
rounded_box(ax, 5.5, 2.3, 3.0, 0.85, C_GOLD,
            "SageMaker Model Registry",
            "vaultech-bath-predictor-group\nMAE=0.92s · R²=0.69 · Approved")

# 7. Gold parquet (inside ECS box, indicated by label)
rounded_box(ax, 5.5, 1.0, 2.6, 0.65, "#78909C",
            "Gold Parquet (bundled in image)",
            "data/gold/pieces.parquet  169 k pieces", fontsize=8)


# ── arrows ────────────────────────────────────────────────────────────────────

# User ↔ ECS
arrow(ax, 2.2, 4.6, 4.0, 4.6, "HTTP request", C_BLUE)
arrow(ax, 4.0, 4.4, 2.2, 4.4, "HTML/prediction", C_BLUE, style="<-")

# ECR → ECS  (image pull at deploy time)
curved_arrow(ax, 4.5, 5.85, 5.2, 5.05, "pull image", C_GRAY, rad=-0.2)

# ECS → SageMaker
arrow(ax, 7.0, 4.5, 8.7, 4.5, "invoke_endpoint\n(CSV payload)", C_PURPLE)
arrow(ax, 8.7, 4.35, 7.0, 4.35, "float prediction", C_PURPLE, style="<-")

# SageMaker → S3  (load model on startup)
arrow(ax, 10.2, 3.95, 10.2, 2.75, "load model.tar.gz\n(at startup)", C_GREEN)

# S3 ← deploy script  (small note)
ax.text(8.0, 2.3, "deploy/deploy_sagemaker.py\nuploads + registers", ha="center",
        va="center", fontsize=7.5, color="#546E7A", style="italic")
arrow(ax, 8.85, 2.3, 8.7, 2.3, "", C_GREEN)

# Model Registry ↔ SageMaker
curved_arrow(ax, 7.0, 2.3, 8.7, 3.95, "registered\nmodel package", C_GOLD, rad=0.2)

# ECS reads gold parquet
arrow(ax, 5.5, 3.95, 5.5, 1.35, "reads piece history", "#78909C")


# ── legend / notes ────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_BLUE,   label="User / Browser"),
    mpatches.Patch(color=C_TEAL,   label="ECS / Fargate (Streamlit)"),
    mpatches.Patch(color=C_PURPLE, label="SageMaker Endpoint"),
    mpatches.Patch(color=C_GREEN,  label="Amazon S3"),
    mpatches.Patch(color=C_GOLD,   label="Model Registry"),
    mpatches.Patch(color=C_GRAY,   label="Amazon ECR"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          framealpha=0.9, edgecolor=C_BORDER, bbox_to_anchor=(0.0, 0.0))


plt.tight_layout()
plt.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {OUTPUT}")
