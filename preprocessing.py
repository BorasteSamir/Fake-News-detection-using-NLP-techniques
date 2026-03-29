"""
preprocessing.py
================
Modular NLP preprocessing pipeline for text classification.

Steps:
    1. Lowercasing
    2. Remove URLs, HTML tags
    3. Remove punctuation, numbers, special characters
    4. Tokenization
    5. Stopword removal
    6. Lemmatization
    7. Class imbalance detection and handling (SMOTE-ready)
"""

import re
import string
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (only once)
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# Initialize reusable objects
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ── Individual Cleaning Functions ──────────────────────────────────────────

def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_html_tags(text: str) -> str:
    """Strip HTML tags."""
    return re.sub(r"<.*?>", " ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove digits and numeric tokens."""
    return re.sub(r"\d+", " ", text)


def remove_special_characters(text: str) -> str:
    """Remove non-alphabetic characters (keeps spaces)."""
    return re.sub(r"[^a-z\s]", " ", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces into one and strip leading/trailing."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    """Tokenize text into word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove common English stopwords."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def lemmatize(tokens: list) -> list:
    """Apply lemmatization to reduce tokens to base form."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


# ── Main Pipeline Function ─────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline applied to a single text string.

    Pipeline:
        lowercase → remove URLs → remove HTML → remove punctuation →
        remove numbers → remove special chars → tokenize →
        remove stopwords → lemmatize → rejoin

    Args:
        text: Raw text string

    Returns:
        Clean, preprocessed text string
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)


def preprocess_series(series: pd.Series, verbose: bool = True) -> pd.Series:
    """
    Apply preprocess_text() to a pandas Series.

    Args:
        series: pandas Series of raw text
        verbose: Print progress

    Returns:
        pandas Series of cleaned text
    """
    if verbose:
        print(f"[INFO] Preprocessing {len(series)} text samples...")

    cleaned = series.apply(preprocess_text)

    if verbose:
        empty_count = (cleaned.str.strip() == "").sum()
        print(f"[INFO] Preprocessing complete. Empty results: {empty_count}")

    return cleaned


# ── Class Imbalance Detection ──────────────────────────────────────────────

def check_class_imbalance(y: pd.Series, threshold: float = 1.5) -> dict:
    """
    Detect class imbalance in label distribution.

    Args:
        y: Label Series
        threshold: Ratio above which imbalance is flagged

    Returns:
        Dictionary with class counts, ratio, and imbalance flag
    """
    counts = y.value_counts().sort_index()
    majority = counts.max()
    minority = counts.min()
    ratio = majority / minority

    is_imbalanced = ratio > threshold

    result = {
        "counts": counts.to_dict(),
        "majority_class": counts.idxmax(),
        "minority_class": counts.idxmin(),
        "imbalance_ratio": round(ratio, 3),
        "is_imbalanced": is_imbalanced
    }

    print(f"\n[INFO] Class Distribution: {result['counts']}")
    print(f"[INFO] Imbalance Ratio   : {result['imbalance_ratio']:.3f}")

    if is_imbalanced:
        print(f"[WARNING] Dataset is imbalanced (ratio > {threshold}).")
        print("          Consider applying SMOTE or class_weight='balanced' in models.")
    else:
        print("[INFO] Dataset is approximately balanced. No oversampling needed.")

    return result
