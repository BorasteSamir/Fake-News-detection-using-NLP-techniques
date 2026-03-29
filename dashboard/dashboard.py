"""
dashboard.py
============
Simple Streamlit Web Dashboard for Fake News Detection

Features:
- Text input for manual news entry
- Real-time prediction using trained models
- Confidence scores

Run with: streamlit run dashboard.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Import your preprocessing functions
from preprocess import preprocess_text

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .real-news {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .fake-news {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .confidence-score {
        font-size: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained SVM model and TF-IDF vectorizer"""
    try:
        # Try to load saved model and vectorizer
        with open('../models/svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please train the model first by running main.py")
        return None, None

def predict_news(text, model, vectorizer):
    """Make prediction on the input text"""
    if not text or not text.strip():
        return None, None
    
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text or not cleaned_text.strip():
        return None, None
    
    # Vectorize the text
    text_vector = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    confidence = model.predict_proba(text_vector)[0]
    
    return prediction, confidence

def main():
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("📋 How to Use")
    st.sidebar.markdown("""
    1. **Paste Text**: Enter news article text in the input box
    2. **Click Analyze** to get prediction
    
    The system will classify the news as:
    - ✅ **Real News** (Legitimate)
    - ❌ **Fake News** (Potentially misleading)
    """)
    
    st.sidebar.header("📊 Model Info")
    st.sidebar.info("""
    **Model**: Support Vector Machine (SVM)
    **Features**: TF-IDF Vectorization
    **Accuracy**: ~99.2%
    **Training Data**: 44,000+ news articles
    """)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Input News Content")
        
        text_input = st.text_area(
            "Paste news article text here:",
            height=300,
            placeholder="Enter the news article text you want to analyze..."
        )
    
    with col2:
        st.header("🎯 Analysis Results")
        
        if st.button("🔍 Analyze News", type="primary", use_container_width=True):
            if text_input:
                with st.spinner("Analyzing..."):
                    prediction, confidence = predict_news(text_input, model, vectorizer)
                    
                    if prediction is not None:
                        # Display prediction
                        if prediction == 0:  # Real News
                            st.markdown(f"""
                            <div class="prediction-box real-news">
                                ✅ REAL NEWS
                            </div>
                            """, unsafe_allow_html=True)
                            confidence_score = confidence[0] * 100
                        else:  # Fake News
                            st.markdown(f"""
                            <div class="prediction-box fake-news">
                                ❌ FAKE NEWS
                            </div>
                            """, unsafe_allow_html=True)
                            confidence_score = confidence[1] * 100
                        
                        # Display confidence
                        st.markdown(f"""
                        <div class="confidence-score">
                            <strong>Confidence Score:</strong> {confidence_score:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional info
                        st.subheader("📈 Detailed Analysis")
                        col_real, col_fake = st.columns(2)
                        
                        with col_real:
                            st.metric("Real News Probability", f"{confidence[0]:.3f}")
                        
                        with col_fake:
                            st.metric("Fake News Probability", f"{confidence[1]:.3f}")
                        
                        # Explanation
                        st.subheader("💡 What This Means")
                        if prediction == 0:
                            st.success("""
                            This article appears to be **legitimate news** based on its language patterns, 
                            writing style, and content structure. However, always verify information 
                            from multiple reliable sources.
                            """)
                        else:
                            st.warning("""
                            This article shows characteristics commonly found in **misleading or fake news**. 
                            Please fact-check this information with reliable news sources before sharing.
                            """)
                    
                    else:
                        st.error("❌ Could not analyze the text. Please ensure the input contains meaningful content.")
            
            else:
                st.warning("⚠️ Please provide text input.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 Powered by Machine Learning | Built with Streamlit</p>
        <p><strong>Disclaimer:</strong> This tool provides predictions based on text patterns. 
        Always verify news from multiple reliable sources.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()