"""
visualization.py
================
Generate and save all visualizations:
    1. Confusion matrices  (one per model)
    2. Model performance comparison bar chart
    3. Word clouds         (Fake News vs Real News)
    4. LSTM training history (loss & accuracy curves)

All outputs are saved to the `outputs/` directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = {
    "real"    : "#2ecc71",
    "fake"    : "#e74c3c",
    "primary" : "#2c3e50",
    "accent"  : "#3498db",
}


# ══════════════════════════════════════════════════════════════════════════
#  1. Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    class_names: list = None
) -> None:
    """
    Save a styled confusion matrix heatmap for a single model.

    Args:
        cm          : Confusion matrix array (from sklearn)
        model_name  : Model identifier used in filename and title
        class_names : List of label names (default: ['Real', 'Fake'])
    """
    if class_names is None:
        class_names = ["Real News", "Fake News"]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    path = os.path.join(OUTPUT_DIR, f"cm_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved → {path}")


def plot_all_confusion_matrices(all_results: list) -> None:
    """
    Plot all confusion matrices in a single combined figure (grid layout).

    Args:
        all_results : List of result dicts from evaluation.evaluate_model()
    """
    n = len(all_results)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    class_names = ["Real News", "Fake News"]

    for i, result in enumerate(all_results):
        ax = axes_flat[i]
        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"{result['model']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Confusion Matrices – All Models", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "confusion_matrices_all.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Combined confusion matrix chart saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
#  2. Model Performance Comparison
# ══════════════════════════════════════════════════════════════════════════

def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Grouped bar chart comparing Accuracy, F1 (Macro), and F1 (Weighted)
    across all models.

    Args:
        comparison_df : DataFrame from evaluation.build_comparison_table()
    """
    metrics = ["Accuracy", "F1 (Macro)", "F1 (Weighted)"]
    models  = comparison_df["Model"].tolist()

    x = np.arange(len(models))
    bar_width = 0.25
    colors = ["#3498db", "#9b59b6", "#e67e22"]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = comparison_df[metric].tolist()
        bars = ax.bar(
            x + i * bar_width,
            values,
            bar_width,
            label=metric,
            color=color,
            alpha=0.85,
            edgecolor="white"
        )
        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.003,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold"
            )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.9, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    sns.despine(left=False, bottom=False)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Model comparison chart saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
#  3. Word Clouds
# ══════════════════════════════════════════════════════════════════════════

def plot_wordclouds(df: pd.DataFrame, text_col: str = "content") -> None:
    """
    Generate and save word clouds for Fake and Real news separately.

    Args:
        df       : Full dataset DataFrame
        text_col : Column containing preprocessed text
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[WARNING] wordcloud not installed. Skipping word cloud generation.")
        print("          Install via: pip install wordcloud")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Word Clouds – Fake vs Real News", fontsize=16, fontweight="bold")

    configs = [
        {"label": 0, "title": "Real News",  "colormap": "Greens",  "bg": "#0d1b0d"},
        {"label": 1, "title": "Fake News",  "colormap": "Reds",    "bg": "#1b0d0d"},
    ]

    for ax, cfg in zip(axes, configs):
        text_data = df[df["label"] == cfg["label"]][text_col].dropna()
        corpus    = " ".join(text_data.values)

        wc = WordCloud(
            width=900,
            height=500,
            background_color=cfg["bg"],
            colormap=cfg["colormap"],
            max_words=200,
            collocations=False,
            prefer_horizontal=0.9
        ).generate(corpus)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(cfg["title"], fontsize=14, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "wordclouds.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"[INFO] Word clouds saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
#  4. LSTM Training History
# ══════════════════════════════════════════════════════════════════════════

def plot_lstm_history(history) -> None:
    """
    Plot training vs validation loss and accuracy for the LSTM model.

    Args:
        history : Keras History object returned from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["loss"]) + 1)

    # Loss
    ax1.plot(epochs, history.history["loss"],     "o-",  color="#e74c3c", label="Train Loss")
    ax1.plot(epochs, history.history["val_loss"],  "s--", color="#c0392b", label="Val Loss",   alpha=0.8)
    ax1.set_title("LSTM – Loss Curve",     fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history.history["accuracy"],     "o-",  color="#3498db", label="Train Accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], "s--", color="#2980b9", label="Val Accuracy", alpha=0.8)
    ax2.set_title("LSTM – Accuracy Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Bidirectional LSTM Training History", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "lstm_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] LSTM training history saved → {path}")
