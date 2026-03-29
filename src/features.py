"""
features.py
===========
Multiple text vectorization strategies:
    1. Bag of Words  (CountVectorizer)
    2. TF-IDF        (TfidfVectorizer)
    3. Word2Vec      (Gensim – optional, falls back gracefully)

Each strategy returns (X_train_vec, X_test_vec, vectorizer/model).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ── 1. Bag of Words ────────────────────────────────────────────────────────

def get_bow_features(
    X_train: pd.Series,
    X_test: pd.Series,
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2)
):
    """
    Transform text using Bag-of-Words (CountVectorizer).

    Args:
        X_train       : Training text Series (preprocessed)
        X_test        : Test text Series (preprocessed)
        max_features  : Vocabulary cap
        ngram_range   : Unigrams + bigrams by default

    Returns:
        X_train_bow, X_test_bow, vectorizer
    """
    print("[INFO] Building Bag-of-Words features...")

    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,          # ignore very rare terms
        max_df=0.95        # ignore very common terms
    )

    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow  = vectorizer.transform(X_test)

    print(f"       BoW vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"       Train matrix shape  : {X_train_bow.shape}")
    print(f"       Test  matrix shape  : {X_test_bow.shape}")

    return X_train_bow, X_test_bow, vectorizer


# ── 2. TF-IDF ──────────────────────────────────────────────────────────────

def get_tfidf_features(
    X_train: pd.Series,
    X_test: pd.Series,
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True
):
    """
    Transform text using TF-IDF weighting.

    Args:
        X_train       : Training text Series (preprocessed)
        X_test        : Test text Series (preprocessed)
        max_features  : Vocabulary cap
        ngram_range   : Unigrams + bigrams by default
        sublinear_tf  : Apply log normalization to term frequency

    Returns:
        X_train_tfidf, X_test_tfidf, vectorizer
    """
    print("[INFO] Building TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f"       TF-IDF vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"       Train matrix shape     : {X_train_tfidf.shape}")
    print(f"       Test  matrix shape     : {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer


# ── 3. Word2Vec (Optional) ─────────────────────────────────────────────────

def _text_to_avg_vector(tokens: list, model, vector_size: int) -> np.ndarray:
    """Average Word2Vec vectors for all tokens in a document."""
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def get_word2vec_features(
    X_train: pd.Series,
    X_test: pd.Series,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 10
):
    """
    Train a Word2Vec model on training data and produce averaged document vectors.

    Falls back with a clear error if gensim is not installed.

    Args:
        X_train     : Preprocessed training text Series
        X_test      : Preprocessed test text Series
        vector_size : Embedding dimensionality
        window      : Context window size
        min_count   : Minimum token frequency
        epochs      : Training epochs

    Returns:
        X_train_w2v (np.ndarray), X_test_w2v (np.ndarray), w2v_model
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError(
            "gensim is required for Word2Vec. Install via: pip install gensim"
        )

    print("[INFO] Training Word2Vec model...")

    # Tokenise each document
    train_tokens = [text.split() for text in X_train]
    test_tokens  = [text.split() for text in X_test]

    # Train Word2Vec on training corpus only
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        seed=42
    )

    print(f"       Word2Vec vocab size : {len(w2v_model.wv)}")

    # Build document-level averaged vectors
    X_train_w2v = np.array([
        _text_to_avg_vector(tokens, w2v_model, vector_size)
        for tokens in train_tokens
    ])
    X_test_w2v = np.array([
        _text_to_avg_vector(tokens, w2v_model, vector_size)
        for tokens in test_tokens
    ])

    print(f"       Train matrix shape : {X_train_w2v.shape}")
    print(f"       Test  matrix shape : {X_test_w2v.shape}")

    return X_train_w2v, X_test_w2v, w2v_model