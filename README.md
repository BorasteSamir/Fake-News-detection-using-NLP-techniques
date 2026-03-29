# Fake News Detection

A machine learning pipeline to classify news articles as real or fake using NLP techniques and multiple classification models.

## Features

- Complete NLP preprocessing pipeline (tokenization, stopword removal, lemmatization)
- Multiple vectorization methods (Bag-of-Words, TF-IDF, Word2Vec)
- 5 different classification models with performance comparison
- Comprehensive evaluation metrics and visualizations
- Automated EDA and result generation

## Models Used

- **Logistic Regression** (TF-IDF features)
- **Naive Bayes** (Bag-of-Words features)  
- **Support Vector Machine** (TF-IDF features)
- **Random Forest** (TF-IDF features)
- **BiLSTM Neural Network** (Word embeddings)

## Results

**Best Model:** SVM with TF-IDF features  
**Accuracy:** ~99.2%  
**F1-Score:** ~99.2%

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place dataset files (`fake.csv`, `real.csv`) in project directory

3. Run the pipeline:
   ```bash
   python main.py
   ```

4. Check results in `outputs/` folder

## Project Structure

```
├── main.py                 # Main pipeline orchestrator
├── data_loader.py          # Data loading and EDA
├── preprocessing.py        # Text preprocessing
├── feature_engineering.py  # Vectorization methods
├── models.py              # ML model implementations
├── evaluation.py          # Performance metrics
├── visualization.py       # Charts and plots
├── requirements.txt       # Dependencies
└── outputs/              # Generated results
```

## Author

Samir Boraste