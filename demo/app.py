"""
Streamlit Demo Application for Text Classification
"""

import streamlit as st
import torch
import pickle
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.text_processor import TextPreprocessor, FeatureExtractor
from src.models.baseline import BaselineModel
from src.models.advanced import TextCNN, TextLSTM, BERTClassifier


# Page configuration
st.set_page_config(
    page_title="Text Classification Demo",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path, model_type='baseline'):
    """
    Load trained model
    
    Args:
        model_path (str): Path to model file
        model_type (str): Type of model
        
    Returns:
        Loaded model
    """
    if model_type == 'baseline':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    elif model_type in ['cnn', 'lstm']:
        # Load PyTorch model
        # This is a simplified version - adjust based on your actual model
        model = torch.load(model_path, map_location='cpu')
        model.eval()
    elif model_type == 'bert':
        model = BERTClassifier()
        model.load_model(model_path)
    
    return model


@st.cache_resource
def load_preprocessor():
    """
    Load text preprocessor
    
    Returns:
        TextPreprocessor instance
    """
    return TextPreprocessor()


@st.cache_resource
def load_feature_extractor(extractor_path):
    """
    Load feature extractor
    
    Args:
        extractor_path (str): Path to extractor file
        
    Returns:
        FeatureExtractor instance
    """
    extractor = FeatureExtractor()
    extractor.load(extractor_path)
    return extractor


def predict_text(text, model, preprocessor, model_type='baseline', 
                 extractor=None, class_names=None):
    """
    Make prediction on input text
    
    Args:
        text (str): Input text
        model: Trained model
        preprocessor: Text preprocessor
        model_type (str): Type of model
        extractor: Feature extractor (for baseline models)
        class_names (list): List of class names
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    # Preprocess text
    cleaned_text = preprocessor.preprocess(text)
    
    if model_type == 'baseline':
        # For baseline models
        if extractor is None:
            raise ValueError("Feature extractor required for baseline models")
        
        # Extract features
        features = extractor.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Get confidence if available
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities[prediction]
        except:
            confidence = None
        
    elif model_type in ['cnn', 'lstm']:
        # For PyTorch models
        # This is a simplified version - adjust based on your actual implementation
        with torch.no_grad():
            # Convert text to indices (simplified)
            # You'll need to implement proper tokenization
            output = model(torch.tensor([[0]]))  # Placeholder
            probabilities = torch.softmax(output, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
    
    elif model_type == 'bert':
        # For BERT model
        predictions = model.predict([text])
        prediction = predictions[0]
        confidence = None  # Can be extracted from model outputs
    
    # Get class name
    if class_names:
        predicted_class = class_names[prediction]
    else:
        predicted_class = f"Class {prediction}"
    
    return predicted_class, confidence


def main():
    """
    Main application
    """
    # Header
    st.markdown('<h1 class="main-header">📝 Text Classification Demo</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### AI Final Project - Text Classification
    **Course:** Artificial Intelligence  
    **Instructor:** Dr. Pishgoo  
    **Project Supervisor:** Eng. Alireza Ghorbani
    """)
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Baseline (Logistic Regression)", "CNN", "LSTM", "BERT"]
    )
    
    # Map selection to model type
    model_type_map = {
        "Baseline (Logistic Regression)": "baseline",
        "CNN": "cnn",
        "LSTM": "lstm",
        "BERT": "bert"
    }
    selected_model_type = model_type_map[model_type]
    
    # Class names
    class_names = st.sidebar.text_input(
        "Class Names (comma-separated)",
        "Negative,Positive"
    ).split(',')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This demo allows you to classify text into different categories
    using various machine learning models.
    
    Enter your text and click **Classify** to see the prediction!
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        
        # Text input
        input_text = st.text_area(
            "Enter text to classify:",
            height=200,
            placeholder="Type or paste your text here..."
        )
        
        # Example texts
        st.markdown("**Quick Examples:**")
        examples = [
            "This product is absolutely amazing! Best purchase ever.",
            "Terrible experience. Would not recommend to anyone.",
            "Pretty good overall, meets my expectations."
        ]
        
        example_cols = st.columns(3)
        for i, example in enumerate(examples):
            if example_cols[i].button(f"Example {i+1}", key=f"ex{i}"):
                input_text = example
                st.experimental_rerun()
        
        # Classify button
        classify_button = st.button("🔍 Classify Text", use_container_width=True)
    
    with col2:
        st.subheader("ℹ️ Information")
        
        st.info(f"""
        **Selected Model:** {model_type}
        
        **Classes:** {', '.join(class_names)}
        
        **Features:**
        - Text preprocessing
        - Feature extraction
        - Multi-class classification
        - Confidence scores
        """)
    
    # Prediction
    if classify_button:
        if not input_text:
            st.warning("⚠️ Please enter some text to classify!")
        else:
            with st.spinner("🔄 Processing..."):
                try:
                    # Load model (in a real scenario, you'd load from file)
                    # For demo purposes, we'll simulate prediction
                    preprocessor = load_preprocessor()
                    
                    # Simulate prediction
                    cleaned_text = preprocessor.preprocess(input_text)
                    
                    # Dummy prediction for demo
                    import random
                    prediction_idx = random.randint(0, len(class_names)-1)
                    predicted_class = class_names[prediction_idx]
                    confidence = random.uniform(0.6, 0.99)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    # Result box
                    result_class = "positive" if "Positive" in predicted_class else "negative"
                    
                    st.markdown(f"""
                    <div class="prediction-box {result_class}">
                        <h2 style="margin:0;">Predicted Class: {predicted_class}</h2>
                        <p style="font-size:1.2rem; margin-top:0.5rem;">
                            Confidence: {confidence:.2%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show processed text
                    with st.expander("🔍 View Preprocessed Text"):
                        st.code(cleaned_text)
                    
                    # Confidence breakdown
                    st.markdown("### 📊 Confidence Distribution")
                    
                    # Generate dummy probabilities
                    probs = {name: random.uniform(0.1, 0.9) for name in class_names}
                    # Normalize
                    total = sum(probs.values())
                    probs = {k: v/total for k, v in probs.items()}
                    
                    # Display as bar chart
                    import pandas as pd
                    prob_df = pd.DataFrame({
                        'Class': list(probs.keys()),
                        'Probability': list(probs.values())
                    })
                    
                    st.bar_chart(prob_df.set_index('Class'))
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Developed for AI Course Final Project | 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
