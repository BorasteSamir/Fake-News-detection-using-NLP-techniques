"""
evaluation.py
=============
Compute and report model evaluation metrics:
    - Accuracy
    - Precision, Recall, F1-score (macro + weighted)
    - Confusion Matrix
    - Classification Report

Also provides a helper to build the final comparative results table.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
    model_name: str,
    y_true,
    y_pred,
    verbose: bool = True
) -> dict:
    """
    Compute evaluation metrics for a single model.

    Args:
        model_name : Display name of the model
        y_true     : Ground-truth labels
        y_pred     : Predicted labels
        verbose    : Print results to console

    Returns:
        Dictionary containing all metrics
    """
    acc       = accuracy_score(y_true, y_pred)
    prec_mac  = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_mac   = recall_score(y_true, y_pred,    average="macro",    zero_division=0)
    f1_mac    = f1_score(y_true, y_pred,        average="macro",    zero_division=0)
    prec_wt   = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_wt    = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1_wt     = f1_score(y_true, y_pred,        average="weighted", zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)
    report    = classification_report(
        y_true, y_pred,
        target_names=["Real News", "Fake News"],
        zero_division=0
    )

    results = {
        "model"          : model_name,
        "accuracy"       : round(acc,      4),
        "precision_macro": round(prec_mac, 4),
        "recall_macro"   : round(rec_mac,  4),
        "f1_macro"       : round(f1_mac,   4),
        "precision_wtd"  : round(prec_wt,  4),
        "recall_wtd"     : round(rec_wt,   4),
        "f1_weighted"    : round(f1_wt,    4),
        "confusion_matrix": cm,
        "classification_report": report
    }

    if verbose:
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Model : {model_name}")
        print(sep)
        print(f"  Accuracy          : {acc:.4f}")
        print(f"  Precision (macro) : {prec_mac:.4f}")
        print(f"  Recall    (macro) : {rec_mac:.4f}")
        print(f"  F1-score  (macro) : {f1_mac:.4f}")
        print(f"  F1-score  (wt'd)  : {f1_wt:.4f}")
        print(f"\n  Confusion Matrix:\n{cm}")
        print(f"\n  Classification Report:\n{report}")

    return results


def build_comparison_table(all_results: list) -> pd.DataFrame:
    """
    Aggregate per-model results into a comparison DataFrame.

    Args:
        all_results : List of result dicts from evaluate_model()

    Returns:
        pandas DataFrame sorted by F1-weighted score (descending)
    """
    rows = []
    for r in all_results:
        rows.append({
            "Model"            : r["model"],
            "Accuracy"         : r["accuracy"],
            "Precision (Macro)": r["precision_macro"],
            "Recall (Macro)"   : r["recall_macro"],
            "F1 (Macro)"       : r["f1_macro"],
            "F1 (Weighted)"    : r["f1_weighted"],
        })

    df = pd.DataFrame(rows).sort_values("F1 (Weighted)", ascending=False).reset_index(drop=True)

    # Rank column
    df.insert(0, "Rank", range(1, len(df) + 1))

    return df


def print_comparison_table(comparison_df: pd.DataFrame) -> None:
    """Pretty-print the comparison table to console."""
    sep = "=" * 85
    print(f"\n{sep}")
    print("  MODEL COMPARISON TABLE")
    print(sep)
    print(comparison_df.to_string(index=False))
    print(sep)
