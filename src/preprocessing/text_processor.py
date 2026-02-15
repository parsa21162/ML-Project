"""
Text Preprocessing Module for NLP Tasks
This module handles all text preprocessing operations including:
- Cleaning
- Tokenization
- Stopword removal
- Lemmatization
- Encoding
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for NLP tasks
    """
    
    def __init__(self, language='english'):
        """
        Initialize the preprocessor
        
        Args:
            language (str): Language for stopwords (default: 'english')
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(language))
        
        try:
            word_tokenize("test")
        except LookupError:
            nltk.download('punkt')
        
        try:
            self.lemmatizer.lemmatize("test")
        except LookupError:
            nltk.download('wordnet')
    
    def clean_text(self, text):
        """
        Clean text by removing URLs, mentions, hashtags, and special characters
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [word for word in tokens if word not in self.stop_words]
    
    def lemmatize(self, tokens):
        """
        Lemmatize tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def preprocess(self, text, remove_stopwords=True, lemmatize=True):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize
            
        Returns:
            str: Preprocessed text
        """
        # Clean
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column, target_column=None):
        """
        Preprocess entire dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            target_column (str): Name of target column (optional)
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("Starting preprocessing...")
        
        # Create a copy
        df_processed = df.copy()
        
        # Apply preprocessing
        df_processed['cleaned_text'] = df_processed[text_column].apply(
            lambda x: self.preprocess(x)
        )
        
        # Remove empty texts
        df_processed = df_processed[df_processed['cleaned_text'].str.len() > 0]
        
        print(f"Preprocessing complete. {len(df_processed)} samples remaining.")
        
        return df_processed


class FeatureExtractor:
    """
    Feature extraction class for converting text to numerical features
    """
    
    def __init__(self, method='tfidf', max_features=5000):
        """
        Initialize feature extractor
        
        Args:
            method (str): 'tfidf' or 'bow' (bag of words)
            max_features (int): Maximum number of features
        """
        self.method = method
        self.max_features = max_features
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # unigrams and bigrams
                min_df=2,
                max_df=0.95
            )
        elif method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError("Method must be 'tfidf' or 'bow'")
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts
        
        Args:
            texts (list or pd.Series): Text data
            
        Returns:
            scipy.sparse matrix: Feature matrix
        """
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts (list or pd.Series): Text data
            
        Returns:
            scipy.sparse matrix: Feature matrix
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """
        Get feature names
        
        Returns:
            list: Feature names
        """
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath):
        """
        Save vectorizer
        
        Args:
            filepath (str): Path to save file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to {filepath}")
    
    def load(self, filepath):
        """
        Load vectorizer
        
        Args:
            filepath (str): Path to load file
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"Vectorizer loaded from {filepath}")


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Features
        y: Labels
        test_size (float): Test set proportion
        val_size (float): Validation set proportion
        random_state (int): Random seed
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation from train
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_texts = [
        "This is a great product! I love it.",
        "Terrible service. Would not recommend.",
        "Amazing quality and fast delivery!",
        "Worst purchase ever. Very disappointed."
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess texts
    for text in sample_texts:
        cleaned = preprocessor.preprocess(text)
        print(f"Original: {text}")
        print(f"Cleaned: {cleaned}\n")
    
    # Feature extraction
    extractor = FeatureExtractor(method='tfidf', max_features=100)
    features = extractor.fit_transform(sample_texts)
    print(f"Feature matrix shape: {features.shape}")
