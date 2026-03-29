"""
data_loader.py
==============
Handles loading of fake.csv and real.csv, label assignment,
dataset merging, train-test splitting, and exploratory data analysis (EDA).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ── Output directory for saving plots ──────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(fake_path: str = r"c:\Users\lenovo\Downloads\fake and real news project\fake.csv", real_path: str = r"c:\Users\lenovo\Downloads\fake and real news project\real.csv") -> pd.DataFrame:
    """
    Load fake and real news CSV files, assign labels, and merge into one DataFrame.

    Labels:
        0 → Real News
        1 → Fake News

    Args:
        fake_path: Path to fake.csv
        real_path: Path to real.csv

    Returns:
        Merged DataFrame with columns: title, text, subject, date, label
    """
    print("[INFO] Loading dataset...")

    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"File not found: {fake_path}")
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"File not found: {real_path}")

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    # Assign labels
    fake_df["label"] = 1   # Fake News
    real_df["label"] = 0   # Real News

    # Merge datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)

    # Combine title + text into a single content column for NLP
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Drop rows where content is empty
    df = df[df["content"].str.strip() != ""].reset_index(drop=True)

    print(f"[INFO] Total samples loaded: {len(df)}")
    print(f"       Fake News : {(df['label'] == 1).sum()}")
    print(f"       Real News : {(df['label'] == 0).sum()}")

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified 80/20 train-test split.

    Args:
        df: Merged DataFrame
        test_size: Proportion for test set (default 0.2)
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\n[INFO] Train samples : {len(X_train)}")
    print(f"[INFO] Test  samples : {len(X_test)}")

    return X_train, X_test, y_train, y_test


def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform Exploratory Data Analysis and save plots.

    Generates:
        - Class distribution bar chart
        - Text length distribution (by class)
        - Subject distribution
        - Basic dataset summary printed to console
    """
    print("\n" + "=" * 50)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # ── Basic Stats ────────────────────────────────────────────────────────
    print(f"\nDataset shape   : {df.shape}")
    print(f"Columns         : {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nLabel distribution:\n{df['label'].value_counts().rename({0: 'Real', 1: 'Fake'})}")

    # Add text_length column for analysis
    df = df.copy()
    df["text_length"] = df["content"].str.split().str.len()

    print(f"\nText length statistics (word count):")
    print(df.groupby("label")["text_length"].describe().rename(index={0: "Real", 1: "Fake"}))

    # ── Plot 1: Class Distribution ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Exploratory Data Analysis – Fake News Detection", fontsize=14, fontweight="bold")

    label_counts = df["label"].value_counts().sort_index()
    axes[0].bar(
        ["Real News", "Fake News"],
        label_counts.values,
        color=["#2ecc71", "#e74c3c"],
        edgecolor="black",
        alpha=0.85
    )
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(label_counts.values):
        axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")

    # ── Plot 2: Text Length Distribution ──────────────────────────────────
    real_lengths = df[df["label"] == 0]["text_length"]
    fake_lengths = df[df["label"] == 1]["text_length"]

    axes[1].hist(real_lengths, bins=50, alpha=0.6, color="#2ecc71", label="Real", density=True)
    axes[1].hist(fake_lengths, bins=50, alpha=0.6, color="#e74c3c", label="Fake", density=True)
    axes[1].set_title("Text Length Distribution (Word Count)")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].set_xlim(0, 2000)

    # ── Plot 3: Subject Distribution ───────────────────────────────────────
    if "subject" in df.columns:
        subject_counts = df.groupby(["subject", "label"]).size().unstack(fill_value=0)
        subject_counts.columns = ["Real", "Fake"]
        subject_counts.plot(
            kind="bar",
            ax=axes[2],
            color=["#2ecc71", "#e74c3c"],
            edgecolor="black",
            alpha=0.85
        )
        axes[2].set_title("Subject Distribution by Class")
        axes[2].set_xlabel("Subject")
        axes[2].set_ylabel("Count")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].legend(title="Label")
    else:
        axes[2].text(0.5, 0.5, "No 'subject' column", ha="center", va="center")
        axes[2].set_title("Subject Distribution")

    plt.tight_layout()
    eda_path = os.path.join(OUTPUT_DIR, "eda_overview.png")
    plt.savefig(eda_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] EDA chart saved → {eda_path}")
