# 📰 Fake News Detection Dashboard

A simple, interactive web dashboard for detecting fake news using machine learning. Built with Streamlit and powered by a trained SVM model.

## 🌟 Features

- **Manual Text Input**: Paste news article text directly
- **URL Content Extraction**: Automatically extract content from news URLs
- **Real-time Prediction**: Instant classification as Real or Fake news
- **Confidence Scores**: See prediction confidence percentages
- **Clean Interface**: User-friendly web interface
- **High Accuracy**: ~99.2% accuracy using SVM with TF-IDF features

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
python setup_dashboard.py
streamlit run dashboard.py
```

### Option 2: Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r dashboard_requirements.txt
   ```

2. **Train the model:**
   ```bash
   python train_for_dashboard.py
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📋 Prerequisites

- Python 3.8+
- Dataset files: `fake.csv` and `real.csv` (download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset))

## 🎯 How to Use

### Text Input Method
1. Go to the "Manual Text Input" tab
2. Paste your news article text
3. Click "Analyze News"
4. View the prediction and confidence score

### URL Input Method
1. Go to the "URL Input" tab
2. Enter a news article URL
3. Click "Extract Content from URL"
4. Review the extracted content
5. Click "Analyze News" for prediction

## 🔧 Technical Details

- **Backend**: Python with Streamlit
- **Model**: Support Vector Machine (SVM)
- **Features**: TF-IDF Vectorization (50K features, unigrams + bigrams)
- **Preprocessing**: Comprehensive NLP pipeline (tokenization, stopword removal, lemmatization)
- **Web Scraping**: BeautifulSoup for URL content extraction

## 📁 Dashboard Files

```
├── dashboard.py              # Main Streamlit application
├── train_for_dashboard.py    # Model training script
├── setup_dashboard.py        # Automated setup script
├── dashboard_requirements.txt # Dashboard dependencies
├── models/                   # Saved model files (created after training)
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
```

## 🛡️ Disclaimer

This tool provides predictions based on text patterns learned from training data. Always verify news from multiple reliable sources before making decisions based on the predictions.

## 🔍 Example Usage

**Real News Example:**
```
"Scientists at MIT have developed a new battery technology that could revolutionize electric vehicles. The research, published in Nature Energy, shows promising results for faster charging times."
```

**Fake News Example:**
```
"BREAKING: Government officials confirm aliens have landed! Secret documents reveal decades of cover-up. Click here for shocking details!"
```

## 🚨 Troubleshooting

**Model files not found:**
- Run `python train_for_dashboard.py` to create model files

**URL extraction fails:**
- Some websites block automated content extraction
- Try copying the text manually instead

**Low accuracy on specific content:**
- The model is trained on news articles; other text types may not work well
- Ensure sufficient text length for accurate prediction

## 📊 Model Performance

- **Training Accuracy**: ~99.4%
- **Test Accuracy**: ~99.2%
- **Model Type**: Linear SVM with TF-IDF
- **Training Data**: 44,000+ news articles

---

Built with ❤️ using Streamlit and scikit-learn