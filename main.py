"""
main.py
=======
Main orchestrator for the Fake News Detection pipeline.

This script runs the complete end-to-end pipeline:
1. Data loading and EDA
2. Text preprocessing
3. Feature engineering
4. Model training and evaluation
5. Visualization and comparison
6. Model saving for dashboard

Run from terminal:
    python main.py
"""

import os
import sys
import warnings
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import project modules
from data_loader import load_data, split_data, perform_eda
from preprocess import preprocess_series, check_class_imbalance
from features import get_bow_features, get_tfidf_features
from train import (
    train_logistic_regression, train_naive_bayes, train_svm, 
    train_random_forest, train_lstm, evaluate_model, 
    build_comparison_table, print_comparison_table,
    plot_confusion_matrix, plot_all_confusion_matrices,
    plot_model_comparison, plot_wordclouds, plot_lstm_history,
    save_best_model, banner
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    start_total = time.time()

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 1 – Load Data & EDA
    # ══════════════════════════════════════════════════════════════════════
    banner("STAGE 1 – DATA LOADING & EXPLORATORY DATA ANALYSIS")

    df = load_data()
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
    X_test_clean = preprocess_series(X_test_raw, verbose=False)

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

    # Save best model for dashboard (SVM typically performs best)
    save_best_model(svm_model, tfidf_vectorizer, "svm")

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 6 – Visualizations
    # ══════════════════════════════════════════════════════════════════════
    banner("STAGE 6 – SAVING VISUALIZATIONS")

    plot_all_confusion_matrices(all_results)
    plot_model_comparison(comparison_df)

    # ══════════════════════════════════════════════════════════════════════
    #  DONE
    # ══════════════════════════════════════════════════════════════════════
    elapsed = round(time.time() - start_total, 1)
    banner(f"PIPELINE COMPLETE — Total time: {elapsed}s")
    print(f"\n  All outputs saved to → ./{OUTPUT_DIR}/")
    print("\n  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    • {f}")
    
    print(f"\n  Models saved to → ./models/")
    print("    • svm_model.pkl")
    print("    • tfidf_vectorizer.pkl")
    
    print(f"\n🚀 To run the dashboard:")
    print("   cd dashboard")
    print("   streamlit run dashboard.py")


if __name__ == "__main__":
    main()