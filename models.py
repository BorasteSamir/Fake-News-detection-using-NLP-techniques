"""
models.py
=========
Defines and trains all classification models:

    Classical ML (TF-IDF / BoW features):
        1. Logistic Regression
        2. Multinomial Naive Bayes
        3. Support Vector Machine (Linear SVC)
        4. Random Forest

    Deep Learning (raw sequences):
        5. Bidirectional LSTM (Keras / TensorFlow)

Each classical model function accepts sparse feature matrices.
The LSTM pipeline handles its own tokenization and embedding internally.
"""

import numpy as np
import os

# ── Scikit-learn models ────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV   # for LinearSVC proba

# ── Suppress TF verbose logging ────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ══════════════════════════════════════════════════════════════════════════
#  1. Logistic Regression
# ══════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train, C: float = 1.0, max_iter: int = 1000):
    """
    Train Logistic Regression with L2 regularisation.

    Args:
        X_train  : Sparse feature matrix (TF-IDF / BoW)
        y_train  : Label array
        C        : Inverse regularisation strength (higher → less regularisation)
        max_iter : Maximum solver iterations

    Returns:
        Trained LogisticRegression model
    """
    print("[INFO] Training Logistic Regression...")
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("       Logistic Regression training complete.")
    return model


# ══════════════════════════════════════════════════════════════════════════
#  2. Multinomial Naive Bayes
# ══════════════════════════════════════════════════════════════════════════

def train_naive_bayes(X_train, y_train, alpha: float = 0.1):
    """
    Train Multinomial Naive Bayes (well-suited for BoW/TF-IDF counts).

    Args:
        X_train : Sparse feature matrix (must be non-negative)
        y_train : Label array
        alpha   : Laplace smoothing parameter

    Returns:
        Trained MultinomialNB model
    """
    print("[INFO] Training Naive Bayes...")
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    print("       Naive Bayes training complete.")
    return model


# ══════════════════════════════════════════════════════════════════════════
#  3. Support Vector Machine (Linear SVC)
# ══════════════════════════════════════════════════════════════════════════

def train_svm(X_train, y_train, C: float = 1.0):
    """
    Train Linear SVM via LinearSVC, wrapped in CalibratedClassifierCV
    so that predict_proba() is available for downstream analysis.

    Args:
        X_train : Sparse feature matrix
        y_train : Label array
        C       : Regularisation parameter

    Returns:
        Trained CalibratedClassifierCV (wrapping LinearSVC) model
    """
    print("[INFO] Training SVM (LinearSVC + calibration)...")
    base_svm = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    )
    # Wrap to enable probability estimates
    model = CalibratedClassifierCV(base_svm, cv=3)
    model.fit(X_train, y_train)
    print("       SVM training complete.")
    return model


# ══════════════════════════════════════════════════════════════════════════
#  4. Random Forest
# ══════════════════════════════════════════════════════════════════════════

def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 200,
    max_depth: int = None,
    max_features: str = "sqrt"
):
    """
    Train Random Forest classifier.

    Args:
        X_train      : Feature matrix (dense or sparse)
        y_train      : Label array
        n_estimators : Number of trees
        max_depth    : Maximum tree depth (None = fully grown)
        max_features : Number of features per split

    Returns:
        Trained RandomForestClassifier model
    """
    print("[INFO] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("       Random Forest training complete.")
    return model


# ══════════════════════════════════════════════════════════════════════════
#  5. Bidirectional LSTM (Deep Learning)
# ══════════════════════════════════════════════════════════════════════════

def build_lstm_model(
    vocab_size: int,
    max_len: int,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    dropout_rate: float = 0.4
):
    """
    Build a Bidirectional LSTM model using Keras functional API.

    Architecture:
        Embedding → Bidirectional LSTM → GlobalMaxPool → Dense → Dropout → Output

    Args:
        vocab_size    : Vocabulary size (from Tokenizer)
        max_len       : Padded sequence length
        embedding_dim : Embedding vector size
        lstm_units    : LSTM hidden units (per direction)
        dropout_rate  : Dropout probability for regularisation

    Returns:
        Compiled Keras model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, Bidirectional, LSTM,
            GlobalMaxPooling1D, Dense, Dropout, SpatialDropout1D
        )
    except ImportError:
        raise ImportError(
            "TensorFlow is required for the LSTM model.\n"
            "Install via: pip install tensorflow"
        )

    model = Sequential([
        Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            input_length=max_len,
            mask_zero=True
        ),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_lstm(
    X_train: "pd.Series",
    X_test: "pd.Series",
    y_train,
    y_test,
    max_features: int = 30_000,
    max_len: int = 300,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    epochs: int = 5,
    batch_size: int = 64
):
    """
    Full LSTM training pipeline including tokenisation, padding, training,
    and returning predictions.

    Args:
        X_train      : Raw (or preprocessed) training text Series
        X_test       : Raw (or preprocessed) test text Series
        y_train      : Training labels
        y_test       : Test labels (used for early stopping validation)
        max_features : Vocabulary cap for Keras Tokenizer
        max_len      : Maximum sequence length (pad / truncate)
        embedding_dim: Word embedding dimensions
        lstm_units   : LSTM hidden units
        epochs       : Training epochs
        batch_size   : Mini-batch size

    Returns:
        model       : Trained Keras model
        y_pred      : Binary predictions on test set (np.ndarray)
        history     : Keras History object
        tokenizer   : Fitted Keras Tokenizer
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        raise ImportError("TensorFlow is required. pip install tensorflow")

    print("[INFO] Training Bidirectional LSTM...")

    # ── Tokenisation ────────────────────────────────────────────────────
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq  = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test_pad  = pad_sequences(X_test_seq,  maxlen=max_len, padding="post", truncating="post")

    vocab_size = min(max_features, len(tokenizer.word_index))
    print(f"       Vocabulary size    : {vocab_size}")
    print(f"       Sequence max length: {max_len}")

    # ── Build model ──────────────────────────────────────────────────────
    model = build_lstm_model(
        vocab_size=vocab_size,
        max_len=max_len,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units
    )
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            verbose=1
        )
    ]

    # ── Training ─────────────────────────────────────────────────────────
    y_train_arr = np.array(y_train)
    y_test_arr  = np.array(y_test)

    history = model.fit(
        X_train_pad, y_train_arr,
        validation_data=(X_test_pad, y_test_arr),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # ── Predictions ──────────────────────────────────────────────────────
    y_prob = model.predict(X_test_pad, batch_size=batch_size).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print("       LSTM training complete.")
    return model, y_pred, history, tokenizer
