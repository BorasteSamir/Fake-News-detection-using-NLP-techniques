"""
train.py
=========
Consolidated training module that combines all model training, evaluation,
and visualization logic for the Fake News Detection project.

This module includes:
- All ML model implementations (LR, NB, SVM, RF, BiLSTM)
- Evaluation metrics and comparison
- Visualization functions
- Model saving functionality
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Scikit-learn models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train, C: float = 1.0, max_iter: int = 1000):
    """Train Logistic Regression with L2 regularisation."""
    print("[INFO] Training Logistic Regression...")
    model = LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs",
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("       Logistic Regression training complete.")
    return model


def train_naive_bayes(X_train, y_train, alpha: float = 0.1):
    """Train Multinomial Naive Bayes."""
    print("[INFO] Training Naive Bayes...")
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    print("       Naive Bayes training complete.")
    return model


def train_svm(X_train, y_train, C: float = 1.0):
    """Train Linear SVM with probability calibration."""
    print("[INFO] Training SVM (LinearSVC + calibration)...")
    base_svm = LinearSVC(
        C=C, class_weight="balanced", max_iter=3000, random_state=42
    )
    model = CalibratedClassifierCV(base_svm, cv=3)
    model.fit(X_train, y_train)
    print("       SVM training complete.")
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 200, 
                       max_depth: int = None, max_features: str = "sqrt"):
    """Train Random Forest classifier."""
    print("[INFO] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("       Random Forest training complete.")
    return model


def build_lstm_model(vocab_size: int, max_len: int, embedding_dim: int = 128,
                    lstm_units: int = 128, dropout_rate: float = 0.4):
    """Build a Bidirectional LSTM model."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, 
            Dense, Dropout, SpatialDropout1D
        )
    except ImportError:
        raise ImportError("TensorFlow is required for LSTM. Install via: pip install tensorflow")

    model = Sequential([
        Embedding(vocab_size + 1, embedding_dim, input_length=max_len, mask_zero=True),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(X_train, X_test, y_train, y_test, max_features: int = 30_000,
              max_len: int = 300, embedding_dim: int = 128, lstm_units: int = 128,
              epochs: int = 5, batch_size: int = 64):
    """Full LSTM training pipeline."""
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        raise ImportError("TensorFlow is required. pip install tensorflow")

    print("[INFO] Training Bidirectional LSTM...")
    
    # Tokenization
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")
    
    vocab_size = min(max_features, len(tokenizer.word_index))
    print(f"       Vocabulary size: {vocab_size}")
    
    # Build and train model
    model = build_lstm_model(vocab_size, max_len, embedding_dim, lstm_units)
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1)
    ]
    
    history = model.fit(
        X_train_pad, np.array(y_train),
        validation_data=(X_test_pad, np.array(y_test)),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1
    )
    
    # Predictions
    y_prob = model.predict(X_test_pad, batch_size=batch_size).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    
    print("       LSTM training complete.")
    return model, y_pred, history, tokenizer


# ══════════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(model_name: str, y_true, y_pred, verbose: bool = True) -> dict:
    """Compute evaluation metrics for a single model."""
    acc = accuracy_score(y_true, y_pred)
    prec_mac = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_mac = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_wt = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_wt = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Real News", "Fake News"], zero_division=0)

    results = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_mac, 4),
        "recall_macro": round(rec_mac, 4),
        "f1_macro": round(f1_mac, 4),
        "precision_wtd": round(prec_wt, 4),
        "recall_wtd": round(rec_wt, 4),
        "f1_weighted": round(f1_wt, 4),
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

    return results


def build_comparison_table(all_results: list) -> pd.DataFrame:
    """Build comparison table from all model results."""
    rows = []
    for r in all_results:
        rows.append({
            "Model": r["model"],
            "Accuracy": r["accuracy"],
            "Precision (Macro)": r["precision_macro"],
            "Recall (Macro)": r["recall_macro"],
            "F1 (Macro)": r["f1_macro"],
            "F1 (Weighted)": r["f1_weighted"],
        })
    
    df = pd.DataFrame(rows).sort_values("F1 (Weighted)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def print_comparison_table(comparison_df: pd.DataFrame) -> None:
    """Pretty-print the comparison table."""
    sep = "=" * 85
    print(f"\n{sep}")
    print("  MODEL COMPARISON TABLE")
    print(sep)
    print(comparison_df.to_string(index=False))
    print(sep)


# ══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, model_name: str):
    """Plot and save confusion matrix for a single model."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real News", "Fake News"],
        yticklabels=["Real News", "Fake News"],
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    
    # Save plot
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    filename = f"cm_{safe_name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved → {filepath}")


def plot_all_confusion_matrices(all_results: list):
    """Plot all confusion matrices in a single figure."""
    n_models = len(all_results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Confusion Matrices - All Models", fontsize=16, fontweight="bold")
    
    for i, result in enumerate(all_results):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        sns.heatmap(
            result["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
            ax=ax, cbar=False
        )
        ax.set_title(result["model"], fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    
    # Hide empty subplots
    for i in range(n_models, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "confusion_matrices_all.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] All confusion matrices saved → {filepath}")


def plot_model_comparison(comparison_df: pd.DataFrame):
    """Plot model comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(comparison_df["Model"], comparison_df["Accuracy"], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(comparison_df))))
    ax1.set_title("Model Accuracy Comparison", fontweight="bold")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.9, 1.0)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score comparison
    bars2 = ax2.bar(comparison_df["Model"], comparison_df["F1 (Weighted)"],
                   color=plt.cm.plasma(np.linspace(0, 1, len(comparison_df))))
    ax2.set_title("Model F1-Score Comparison", fontweight="bold")
    ax2.set_ylabel("F1-Score (Weighted)")
    ax2.set_ylim(0.9, 1.0)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Model comparison chart saved → {filepath}")


def plot_wordclouds(df: pd.DataFrame, text_col: str = "content"):
    """Generate and save word clouds for real vs fake news."""
    real_text = " ".join(df[df["label"] == 0][text_col].astype(str))
    fake_text = " ".join(df[df["label"] == 1][text_col].astype(str))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Real news word cloud
    real_wc = WordCloud(width=800, height=400, background_color="white", 
                       colormap="Greens", max_words=100).generate(real_text)
    ax1.imshow(real_wc, interpolation="bilinear")
    ax1.set_title("Real News - Word Cloud", fontsize=14, fontweight="bold", color="darkgreen")
    ax1.axis("off")
    
    # Fake news word cloud
    fake_wc = WordCloud(width=800, height=400, background_color="white",
                       colormap="Reds", max_words=100).generate(fake_text)
    ax2.imshow(fake_wc, interpolation="bilinear")
    ax2.set_title("Fake News - Word Cloud", fontsize=14, fontweight="bold", color="darkred")
    ax2.axis("off")
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "wordclouds.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Word clouds saved → {filepath}")


def plot_lstm_history(history):
    """Plot LSTM training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "lstm_training_history.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] LSTM training history saved → {filepath}")


# ══════════════════════════════════════════════════════════════════════════
#  MODEL SAVING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def save_best_model(model, vectorizer, model_name: str = "svm"):
    """Save the best performing model and vectorizer for dashboard use."""
    model_path = f"models/{model_name}_model.pkl"
    vectorizer_path = f"models/tfidf_vectorizer.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"[INFO] Best model saved → {model_path}")
    print(f"[INFO] Vectorizer saved → {vectorizer_path}")


def banner(text: str) -> None:
    """Print a section banner to the console."""
    line = "═" * 65
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")