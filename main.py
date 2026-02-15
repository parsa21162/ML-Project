"""
Main Training Script
Complete pipeline for text classification project
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.text_processor import TextPreprocessor, FeatureExtractor, split_data
from src.models.baseline import BaselineModel, ModelComparison
from src.models.advanced import TextCNN, TextLSTM, BERTClassifier
from src.training.trainer import Trainer, TextDataset
from src.evaluation.evaluator import Evaluator
from src.utils.helpers import (
    set_seed, count_parameters, save_config, 
    plot_word_cloud, plot_text_length_distribution, 
    plot_class_distribution, create_vocab_from_texts, 
    text_to_indices
)


def load_data(data_path):
    """
    Load dataset from CSV file
    
    Args:
        data_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


def perform_eda(df, text_column, label_column, class_names=None, save_dir='./results/charts'):
    """
    Perform Exploratory Data Analysis
    
    Args:
        df (pd.DataFrame): Dataframe
        text_column (str): Name of text column
        label_column (str): Name of label column
        class_names (list): List of class names
        save_dir (str): Directory to save plots
    """
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Basic statistics
    print(f"\nDataset size: {len(df)}")
    print(f"Number of classes: {df[label_column].nunique()}")
    print(f"\nClass distribution:")
    print(df[label_column].value_counts())
    
    # Plot class distribution
    plot_class_distribution(
        df[label_column].values,
        class_names=class_names,
        save_path=os.path.join(save_dir, 'class_distribution.png')
    )
    
    # Plot text length distribution
    plot_text_length_distribution(
        df[text_column].tolist(),
        labels=df[label_column].tolist(),
        save_path=os.path.join(save_dir, 'text_length_distribution.png')
    )
    
    # Generate word clouds for each class
    if class_names:
        for i, class_name in enumerate(class_names):
            class_texts = df[df[label_column] == i][text_column].tolist()
            if len(class_texts) > 0:
                plot_word_cloud(
                    class_texts,
                    title=f"Word Cloud - {class_name}",
                    save_path=os.path.join(save_dir, f'wordcloud_{class_name}.png')
                )


def train_baseline_models(X_train, X_val, X_test, y_train, y_val, y_test, save_dir='./results/models'):
    """
    Train and compare baseline models
    
    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Labels
        save_dir (str): Directory to save models
        
    Returns:
        dict: Results for all models
    """
    print("\n" + "=" * 50)
    print("TRAINING BASELINE MODELS")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create model comparison
    comparison = ModelComparison()
    
    # Add models
    comparison.add_model("Logistic Regression", "logistic", C=1.0)
    comparison.add_model("Naive Bayes", "naive_bayes")
    comparison.add_model("Random Forest", "random_forest", n_estimators=100)
    comparison.add_model("SVM", "svm", C=1.0)
    
    # Train all models
    comparison.train_all(X_train, y_train)
    
    # Evaluate on validation set
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    val_results = comparison.evaluate_all(X_val, y_val)
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    test_results = comparison.evaluate_all(X_test, y_test)
    
    # Get best model
    best_name, best_model, best_score = comparison.get_best_model('f1_score')
    print(f"\n{'='*50}")
    print(f"Best Model: {best_name}")
    print(f"F1-Score: {best_score:.4f}")
    print(f"{'='*50}")
    
    # Save best model
    best_model.save(os.path.join(save_dir, f'best_baseline_{best_name}.pkl'))
    
    return test_results


def train_deep_learning_model(model_type, train_texts, val_texts, test_texts,
                               y_train, y_val, y_test, config, save_dir='./results/models'):
    """
    Train deep learning model
    
    Args:
        model_type (str): Type of model ('cnn', 'lstm', 'bert')
        train_texts, val_texts, test_texts: Text data
        y_train, y_val, y_test: Labels
        config (dict): Model configuration
        save_dir (str): Directory to save models
        
    Returns:
        tuple: (model, trainer, history)
    """
    print("\n" + "=" * 50)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if model_type in ['cnn', 'lstm']:
        # Create vocabulary
        print("\nCreating vocabulary...")
        all_texts = train_texts + val_texts + test_texts
        word2idx, idx2word = create_vocab_from_texts(
            all_texts,
            max_vocab_size=config.get('vocab_size', 10000)
        )
        
        # Convert texts to indices
        print("Converting texts to indices...")
        train_indices = [text_to_indices(text, word2idx, config.get('max_length', 128)) 
                        for text in train_texts]
        val_indices = [text_to_indices(text, word2idx, config.get('max_length', 128)) 
                      for text in val_texts]
        test_indices = [text_to_indices(text, word2idx, config.get('max_length', 128)) 
                       for text in test_texts]
        
        # Create datasets
        train_dataset = TextDataset(train_indices, y_train)
        val_dataset = TextDataset(val_indices, y_val)
        test_dataset = TextDataset(test_indices, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32))
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32))
        
        # Create model
        if model_type == 'cnn':
            from src.models.advanced import TextCNN
            model = TextCNN(
                vocab_size=len(word2idx),
                embedding_dim=config.get('embedding_dim', 128),
                num_filters=config.get('num_filters', 100),
                num_classes=config.get('num_classes', 2),
                dropout=config.get('dropout', 0.5)
            )
        elif model_type == 'lstm':
            from src.models.advanced import TextLSTM
            model = TextLSTM(
                vocab_size=len(word2idx),
                embedding_dim=config.get('embedding_dim', 128),
                hidden_dim=config.get('hidden_dim', 128),
                num_classes=config.get('num_classes', 2),
                dropout=config.get('dropout', 0.5),
                bidirectional=config.get('bidirectional', True)
            )
        
        # Print model summary
        print(f"\nModel Parameters: {count_parameters(model):,}")
        
        # Create trainer
        trainer = Trainer(
            model,
            device=device,
            learning_rate=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0001),
            model_type=model_type
        )
        
    elif model_type == 'bert':
        from src.models.advanced import BERTClassifier
        from transformers import BertTokenizer
        
        # Create BERT model
        model = BERTClassifier(
            model_name=config.get('model_name', 'bert-base-uncased'),
            num_classes=config.get('num_classes', 2),
            max_length=config.get('max_length', 128)
        )
        
        # Create datasets with BERT tokenizer
        train_dataset = TextDataset(train_texts, y_train, tokenizer=model.tokenizer, 
                                    max_length=config.get('max_length', 128))
        val_dataset = TextDataset(val_texts, y_val, tokenizer=model.tokenizer,
                                  max_length=config.get('max_length', 128))
        test_dataset = TextDataset(test_texts, y_test, tokenizer=model.tokenizer,
                                   max_length=config.get('max_length', 128))
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 16), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 16))
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 16))
        
        # Create trainer
        trainer = Trainer(
            model,
            device=device,
            learning_rate=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01),
            model_type='bert'
        )
    
    # Train model
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config.get('num_epochs', 10),
        patience=config.get('patience', 3),
        save_path=save_dir
    )
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)
    
    evaluator = Evaluator(
        model if model_type != 'bert' else model,
        device=device,
        class_names=config.get('class_names'),
        model_type=model_type
    )
    
    evaluator.predict(test_loader)
    metrics = evaluator.print_metrics()
    evaluator.print_classification_report()
    
    # Save evaluation results
    eval_dir = os.path.join(save_dir, 'evaluation')
    evaluator.save_results(eval_dir)
    
    # Plot training history
    import json
    with open(os.path.join(save_dir, 'training_history.json'), 'r') as f:
        history = json.load(f)
    
    evaluator.plot_training_history(
        history,
        save_path=os.path.join(eval_dir, 'training_history.png')
    )
    
    return model, trainer, history


def main():
    """
    Main function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Text Classification Training Script')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--text_col', type=str, default='text', help='Name of text column')
    parser.add_argument('--label_col', type=str, default='label', help='Name of label column')
    parser.add_argument('--model', type=str, default='baseline', 
                       choices=['baseline', 'cnn', 'lstm', 'bert'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    df = load_data(args.data)
    
    # Determine class names
    unique_labels = sorted(df[args.label_col].unique())
    class_names = [f"Class_{i}" for i in unique_labels]
    num_classes = len(unique_labels)
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Perform EDA
    perform_eda(df, args.text_col, args.label_col, class_names)
    
    # Preprocessing
    print("\n" + "=" * 50)
    print("PREPROCESSING")
    print("=" * 50)
    
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df, args.text_col, args.label_col)
    
    # Get texts and labels
    texts = df_processed['cleaned_text'].tolist()
    labels = df_processed[args.label_col].values
    
    # Train baseline models
    if args.model == 'baseline':
        # Feature extraction
        print("\nExtracting features...")
        extractor = FeatureExtractor(method='tfidf', max_features=5000)
        X = extractor.fit_transform(texts)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, labels, test_size=0.2, val_size=0.1, random_state=args.seed
        )
        
        # Train baseline models
        results = train_baseline_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Train deep learning model
    else:
        # Split data (text, not features)
        from sklearn.model_selection import train_test_split
        
        # First split: train and test
        train_texts, test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=args.seed, stratify=labels
        )
        
        # Second split: train and validation
        train_texts, val_texts, y_train, y_val = train_test_split(
            train_texts, y_train, test_size=0.125, random_state=args.seed, stratify=y_train
        )
        
        print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Configuration
        config = {
            'num_classes': num_classes,
            'class_names': class_names,
            'vocab_size': 10000,
            'embedding_dim': 128,
            'hidden_dim': 128,
            'num_filters': 100,
            'max_length': 128,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
            'patience': 3,
            'dropout': 0.5,
            'bidirectional': True,
            'model_name': 'bert-base-uncased'
        }
        
        # Train model
        model, trainer, history = train_deep_learning_model(
            args.model,
            train_texts, val_texts, test_texts,
            y_train, y_val, y_test,
            config
        )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
