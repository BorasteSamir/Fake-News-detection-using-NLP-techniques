"""
main.py
=======
Orchestrates the complete Fake News Detection NLP pipeline.

Run from terminal:
    python main.py

Pipeline stages:
    1.  Load Data & EDA
    2.  Preprocessing
    3.  Feature Engineering  (BoW + TF-IDF)
    4.  Model Training       (LR, NB, SVM, RF, BiLSTM)
    5.  Evaluation
    6.  Comparative Analysis
    7.  Visualisation
"""

import os
import warnings
import time

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Project modules ────────────────────────────────────────────────────────
from data_loader       import load_data, split_data, perform_eda
from preprocessing     import preprocess_series, check_class_imbalance
from feature_engineering import get_bow_features, get_tfidf_features
from models import (
    train_logistic_regression,
    train_naive_bayes,
    train_svm,
    train_random_forest,
    train_lstm
)
from evaluation import evaluate_model, build_comparison_table, print_comparison_table
from visualization import (
    plot_confusion_matrix,
    plot_all_confusion_matrices,
    plot_model_comparison,
    plot_wordclouds,
    plot_lstm_history
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def banner(text: str) -> None:
    """Print a section banner to the console."""
    line = "═" * 65
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def main():
    start_total = time.time()

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 1 – Load Data & EDA
    # ══════════════════════════════════════════════════════════════════════
    banner("STAGE 1 – DATA LOADING & EXPLORATORY DATA ANALYSIS")

    df = load_data(fake_path="fake.csv", real_path="real.csv")
    perform_eda(df)

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 2 – Preprocessing
    # ══════════════════════════════════════════════════════════════════════
    banner("STAGE 2 – TEXT PREPROCESSING")

    # Generate word clouds BEFORE heavy preprocessing (richer vocabulary)
    plot_wordclouds(df, text_col="content")

    # Preprocess the combined content column
    df["clean_text"] = preprocess_series(df["content"], verbose=True)

    # Drop empty cleaned rows
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    # Train / Test split
    X_train_raw, X_test_raw, y_train, y_test = split_data(df)

    # Apply preprocessing to split sets
    print("[INFO] Preprocessing train split...")
    X_train_clean = preprocess_series(X_train_raw, verbose=False)
    print("[INFO] Preprocessing test split...")
    X_test_clean  = preprocess_series(X_test_raw,  verbose=False)

    # Class imbalance check
    check_class_imbalance(y_train)

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 3 – Feature Engineering
    # ══════════════════════════════════════════════════════════════════════
    banner("STAGE 3 – FEATURE ENGINEERING")

    # Bag-of-Words
    X_train_bow, X_test_bow, bow_vectorizer = get_bow_features(
        X_train_clean, X_test_clean, max_features=50_000
    )

    # TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = get_tfidf_features(
        X_train_clean, X_test_clean, max_features=50_000
    )

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 4 – Model Training + Evaluation
    # ══════════════════════════════════════════════════════════════════════
    banner("STAGE 4 – MODEL TRAINING & EVALUATION")

    all_results = []

    # ── 4a. Logistic Regression (TF-IDF) ─────────────────────────────────
    t0 = time.time()
    lr_model = train_logistic_regression(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    res_lr = evaluate_model("Logistic Regression (TF-IDF)", y_test, y_pred_lr)
    res_lr["train_time"] = round(time.time() - t0, 2)
    all_results.append(res_lr)
    plot_confusion_matrix(res_lr["confusion_matrix"], "Logistic Regression (TF-IDF)")

    # ── 4b. Naive Bayes (BoW) ─────────────────────────────────────────────
    t0 = time.time()
    nb_model = train_naive_bayes(X_train_bow, y_train)
    y_pred_nb = nb_model.predict(X_test_bow)
    res_nb = evaluate_model("Naive Bayes (BoW)", y_test, y_pred_nb)
    res_nb["train_time"] = round(time.time() - t0, 2)
    all_results.append(res_nb)
    plot_confusion_matrix(res_nb["confusion_matrix"], "Naive Bayes (BoW)")

    # ── 4c. SVM (TF-IDF) ─────────────────────────────────────────────────
    t0 = time.time()
    svm_model = train_svm(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)
    res_svm = evaluate_model("SVM (TF-IDF)", y_test, y_pred_svm)
    res_svm["train_time"] = round(time.time() - t0, 2)
    all_results.append(res_svm)
    plot_confusion_matrix(res_svm["confusion_matrix"], "SVM (TF-IDF)")

    # ── 4d. Random Forest (TF-IDF) ───────────────────────────────────────
    t0 = time.time()
    rf_model = train_random_forest(X_train_tfidf, y_train, n_estimators=200)
    y_pred_rf = rf_model.predict(X_test_tfidf)
    res_rf = evaluate_model("Random Forest (TF-IDF)", y_test, y_pred_rf)
    res_rf["train_time"] = round(time.time() - t0, 2)
    all_results.append(res_rf)
    plot_confusion_matrix(res_rf["confusion_matrix"], "Random Forest (TF-IDF)")

    # ── 4e. Bidirectional LSTM ───────────────────────────────────────────
    t0 = time.time()
    lstm_model, y_pred_lstm, lstm_history, lstm_tokenizer = train_lstm(
        X_train_clean, X_test_clean,
        y_train, y_test,
        max_features=30_000,
        max_len=300,
        epochs=6,
        batch_size=64
    )
    res_lstm = evaluate_model("BiLSTM (Embeddings)", y_test, y_pred_lstm)
    res_lstm["train_time"] = round(time.time() - t0, 2)
    all_results.append(res_lstm)
    plot_confusion_matrix(res_lstm["confusion_matrix"], "BiLSTM (Embeddings)")
    plot_lstm_history(lstm_history)

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 5 – Comparative Analysis
    # ══════════════════════════════════════════════════════════════════════
    banner("STAGE 5 – COMPARATIVE ANALYSIS")

    comparison_df = build_comparison_table(all_results)
    print_comparison_table(comparison_df)

    # Save table to CSV
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Comparison table saved → {csv_path}")

    # Bias-variance commentary
    print("\n── Bias-Variance & Overfitting Analysis ──────────────────────")
    print("""
  Logistic Regression (TF-IDF):
    - Low bias, moderate variance.
    - Regularisation (C parameter) prevents overfitting.
    - Excellent linear separator for high-dimensional TF-IDF space.

  Naive Bayes (BoW):
    - High bias model (strong independence assumption).
    - Very low variance; rarely overfits.
    - Fast but may underfit on complex lexical patterns.

  SVM (Linear TF-IDF):
    - Typically the strongest classical model on text.
    - Low bias, controlled variance via margin maximisation.
    - Generalises well; minimal overfitting on sparse high-dim data.

  Random Forest (TF-IDF):
    - Ensemble reduces variance of individual trees.
    - Can overfit on noisy high-dimensional sparse data if depth unconstrained.
    - Slower than linear models but handles non-linear interactions.

  Bidirectional LSTM:
    - Highest model capacity – captures sequential context.
    - Prone to overfitting without dropout, early stopping, LR scheduling.
    - EarlyStopping + ReduceLROnPlateau callbacks mitigate this.
    - Often achieves near-best accuracy but slower to train.
    """)

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 6 – Visualisations
    # ══════════════════════════════════════════════════════════════════════
    banner("STAGE 6 – SAVING VISUALISATIONS")

    plot_all_confusion_matrices(all_results)
    plot_model_comparison(comparison_df)

    # ══════════════════════════════════════════════════════════════════════
    #  DONE
    # ══════════════════════════════════════════════════════════════════════
    elapsed = round(time.time() - start_total, 1)
    banner(f"PIPELINE COMPLETE — Total time: {elapsed}s")
    print(f"\n  All outputs saved to → ./{OUTPUT_DIR}/\n")
    print("  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    • {f}")


if __name__ == "__main__":
    main()
