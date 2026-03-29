"""
Fake News Detection - Source Package
====================================

This package contains all the core modules for the fake news detection pipeline:

- data_loader: Data loading and exploratory data analysis
- preprocess: Text preprocessing and cleaning
- features: Feature engineering and vectorization
- train: Model training, evaluation, and visualization

Usage:
    from src.data_loader import load_data
    from src.preprocess import preprocess_text
    from src.features import get_tfidf_features
    from src.train import train_svm
"""

__version__ = "1.0.0"
__author__ = "Fake News Detection Team"

# Import main functions for easy access
from .data_loader import load_data, split_data, perform_eda
from .preprocess import preprocess_text, preprocess_series, check_class_imbalance
from .features import get_bow_features, get_tfidf_features, get_word2vec_features
from .train import (
    train_logistic_regression, train_naive_bayes, train_svm, 
    train_random_forest, train_lstm, evaluate_model,
    build_comparison_table, save_best_model
)

__all__ = [
    # Data loading
    'load_data', 'split_data', 'perform_eda',
    
    # Preprocessing
    'preprocess_text', 'preprocess_series', 'check_class_imbalance',
    
    # Feature engineering
    'get_bow_features', 'get_tfidf_features', 'get_word2vec_features',
    
    # Training and evaluation
    'train_logistic_regression', 'train_naive_bayes', 'train_svm',
    'train_random_forest', 'train_lstm', 'evaluate_model',
    'build_comparison_table', 'save_best_model'
]