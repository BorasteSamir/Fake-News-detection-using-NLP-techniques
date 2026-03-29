# 📰 Fake News Detection

A comprehensive machine learning pipeline to classify news articles as real or fake using NLP techniques and multiple classification models.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)

## 🌟 Features

- **Complete NLP Pipeline**: Tokenization, stopword removal, lemmatization
- **Multiple Vectorization Methods**: Bag-of-Words, TF-IDF, Word2Vec
- **5 Classification Models**: Logistic Regression, Naive Bayes, SVM, Random Forest, BiLSTM
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrices
- **Rich Visualizations**: EDA plots, word clouds, model comparisons
- **Web Dashboard**: Interactive Streamlit application for real-time predictions
- **Modular Architecture**: Clean, maintainable code structure

## 🏗️ Project Structure

```
fake-news-detection/
│
├── data/                    # Dataset files (download required)
│   ├── Fake.csv            # Fake news dataset
│   └── Real.csv            # Real news dataset
│
├── src/                     # Source code modules
│   ├── __init__.py         # Package initialization
│   ├── data_loader.py      # Data loading and EDA
│   ├── preprocess.py       # Text preprocessing pipeline
│   ├── features.py         # Feature engineering
│   └── train.py            # Model training and evaluation
│
├── dashboard/               # Web dashboard
│   ├── dashboard.py        # Streamlit application
│   └── README.md          # Dashboard documentation
│
├── models/                  # Saved model files (created after training)
├── outputs/                 # Generated results (created after training)
├── main.py                  # Main pipeline orchestrator
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Dataset

Download the dataset from [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place the files in the `data/` directory:

- `data/Fake.csv`
- `data/Real.csv`

**Note**: The dataset files are not included in this repository due to size constraints.

### 4. Run the Complete Pipeline

```bash
python main.py
```

This will:
- Load and analyze the data
- Preprocess the text
- Train all 5 models
- Generate visualizations
- Save the best model for the dashboard

### 5. Launch the Web Dashboard

```bash
cd dashboard
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Model Performance

| Rank | Model | Accuracy | F1-Score | Features |
|------|-------|----------|----------|----------|
| 1 | **SVM (TF-IDF)** | ~99.2% | ~99.2% | TF-IDF Vectorization |
| 2 | Logistic Regression | ~98.9% | ~98.9% | TF-IDF Vectorization |
| 3 | BiLSTM | ~98.7% | ~98.7% | Word Embeddings |
| 4 | Random Forest | ~97.5% | ~97.5% | TF-IDF Vectorization |
| 5 | Naive Bayes | ~94.2% | ~94.2% | Bag-of-Words |

## 🔧 Module Overview

### Core Modules

- **`src/data_loader.py`**: Data loading, preprocessing, and EDA
- **`src/preprocess.py`**: NLP text preprocessing pipeline
- **`src/features.py`**: Feature engineering (BoW, TF-IDF, Word2Vec)
- **`src/train.py`**: Model training, evaluation, and visualization

### Dashboard

- **`dashboard/dashboard.py`**: Interactive Streamlit web application
- Real-time news classification
- Clean, user-friendly interface

## 📈 Generated Outputs

The pipeline generates comprehensive outputs:

- **EDA Overview**: Dataset statistics and distributions
- **Word Clouds**: Visual representation of vocabulary patterns
- **Confusion Matrices**: Model performance visualization
- **Model Comparison**: Accuracy and F1-score comparisons
- **Training History**: Deep learning model progress

## 🎯 Usage Examples

### Basic Usage

```python
from src.data_loader import load_data
from src.preprocess import preprocess_text
from src.features import get_tfidf_features
from src.train import train_svm

# Load and preprocess data
df = load_data()
text = preprocess_text("Your news article text here")

# Train model
model = train_svm(X_train, y_train)
```

### Dashboard Usage

1. **Text Input**: Paste news article text
2. **Analysis**: Click "Analyze News"
3. **Results**: View prediction with confidence scores

## 🛠️ Development

### Adding New Models

1. Add model function to `src/train.py`
2. Update the main pipeline in `main.py`
3. Include in evaluation and comparison

### Customizing Preprocessing

Modify functions in `src/preprocess.py`:
- Add new cleaning steps
- Adjust tokenization parameters
- Modify stopword lists

### Testing

```bash
python test_project.py
```

## 📋 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- nltk, wordcloud, gensim
- tensorflow (for BiLSTM)
- matplotlib, seaborn
- streamlit (for dashboard)

See `requirements.txt` for complete dependency list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Built with**: scikit-learn, TensorFlow, Streamlit
- **Inspired by**: The need for reliable news verification tools

## ⚠️ Disclaimer

This tool provides predictions based on text patterns learned from training data. Always verify news from multiple reliable sources before making decisions based on the predictions.

---

**Made with ❤️ for combating misinformation**