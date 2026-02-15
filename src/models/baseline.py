"""
Baseline Models for Text Classification
This module implements traditional machine learning models:
- Logistic Regression
- Naive Bayes
- Random Forest
- SVM
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class BaselineModel:
    """
    Base class for baseline models
    """
    
    def __init__(self, model_type='logistic', **kwargs):
        """
        Initialize baseline model
        
        Args:
            model_type (str): Type of model ('logistic', 'naive_bayes', 'random_forest', 'svm')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.is_trained = False
    
    def _create_model(self, **kwargs):
        """
        Create the specified model
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            Initialized model
        """
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                **kwargs
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(**kwargs)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        elif self.model_type == 'svm':
            return LinearSVC(
                random_state=42,
                max_iter=1000,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete!")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities (if supported)
        
        Args:
            X: Features
            
        Returns:
            np.array: Prediction probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For SVM
            return self.model.decision_function(X)
        else:
            raise AttributeError(f"{self.model_type} does not support probability predictions")
    
    def evaluate(self, X, y):
        """
        Evaluate the model
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return metrics
    
    def save(self, filepath):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model from file
        
        Args:
            filepath (str): Path to model file
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


class ModelComparison:
    """
    Class to compare multiple baseline models
    """
    
    def __init__(self):
        """
        Initialize model comparison
        """
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model_type, **kwargs):
        """
        Add a model to comparison
        
        Args:
            name (str): Model name
            model_type (str): Type of model
            **kwargs: Model parameters
        """
        self.models[name] = BaselineModel(model_type, **kwargs)
        print(f"Added {name} model")
    
    def train_all(self, X_train, y_train):
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print('='*50)
            model.train(X_train, y_train)
    
    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Results for all models
        """
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {name}")
            print('='*50)
            
            metrics = model.evaluate(X_test, y_test)
            self.results[name] = metrics
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return self.results
    
    def get_best_model(self, metric='f1_score'):
        """
        Get the best performing model
        
        Args:
            metric (str): Metric to compare
            
        Returns:
            tuple: (model_name, model_object, score)
        """
        if not self.results:
            raise RuntimeError("Models must be evaluated first")
        
        best_name = max(self.results, key=lambda x: self.results[x][metric])
        best_score = self.results[best_name][metric]
        
        return best_name, self.models[best_name], best_score


# Example usage
if __name__ == "__main__":
    # This is a placeholder - actual usage would require real data
    print("Baseline Models Module")
    print("=" * 50)
    
    # Example of creating models
    comparison = ModelComparison()
    comparison.add_model("Logistic Regression", "logistic", C=1.0)
    comparison.add_model("Naive Bayes", "naive_bayes")
    comparison.add_model("Random Forest", "random_forest", n_estimators=100)
    comparison.add_model("SVM", "svm", C=1.0)
    
    print("\nAll models initialized successfully!")
