"""
Evaluation Module for Text Classification
Provides comprehensive metrics and visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
import torch
from torch.utils.data import DataLoader
import json
import os


class Evaluator:
    """
    Comprehensive model evaluation class
    """
    
    def __init__(self, model, device=None, class_names=None, model_type='cnn'):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            device: torch device
            class_names (list): List of class names
            model_type (str): Type of model
        """
        self.model = model
        self.model_type = model_type
        self.class_names = class_names
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if model_type != 'bert':
            self.model.to(self.device)
        
        self.predictions = None
        self.true_labels = None
        self.probabilities = None
    
    def predict(self, dataloader):
        """
        Make predictions on dataset
        
        Args:
            dataloader: DataLoader
            
        Returns:
            tuple: (predictions, true_labels, probabilities)
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                if self.model_type == 'bert':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                else:
                    inputs = batch['input'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    logits = self.model(inputs)
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        self.predictions = np.array(all_predictions)
        self.true_labels = np.array(all_labels)
        self.probabilities = np.array(all_probs)
        
        return self.predictions, self.true_labels, self.probabilities
    
    def calculate_metrics(self, average='weighted'):
        """
        Calculate evaluation metrics
        
        Args:
            average (str): Averaging method for multi-class
            
        Returns:
            dict: Dictionary of metrics
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Must run predict() first")
        
        metrics = {
            'accuracy': accuracy_score(self.true_labels, self.predictions),
            'precision': precision_score(
                self.true_labels, self.predictions, average=average, zero_division=0
            ),
            'recall': recall_score(
                self.true_labels, self.predictions, average=average, zero_division=0
            ),
            'f1_score': f1_score(
                self.true_labels, self.predictions, average=average, zero_division=0
            )
        }
        
        # Add per-class metrics if multi-class
        num_classes = len(np.unique(self.true_labels))
        if num_classes > 2:
            for metric_name in ['precision', 'recall', 'f1_score']:
                metric_func = {
                    'precision': precision_score,
                    'recall': recall_score,
                    'f1_score': f1_score
                }[metric_name]
                
                per_class = metric_func(
                    self.true_labels, self.predictions, 
                    average=None, zero_division=0
                )
                
                for i, score in enumerate(per_class):
                    class_name = self.class_names[i] if self.class_names else f"Class_{i}"
                    metrics[f'{metric_name}_{class_name}'] = score
        
        return metrics
    
    def print_metrics(self):
        """
        Print evaluation metrics
        """
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 50)
        print("EVALUATION METRICS")
        print("=" * 50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("=" * 50)
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None, figsize=(10, 8)):
        """
        Plot confusion matrix
        
        Args:
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Must run predict() first")
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else 'auto',
            yticklabels=self.class_names if self.class_names else 'auto'
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path=None, figsize=(10, 8)):
        """
        Plot ROC curve (for binary classification)
        
        Args:
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        if self.probabilities is None:
            raise ValueError("Must run predict() first")
        
        num_classes = self.probabilities.shape[1]
        
        plt.figure(figsize=figsize)
        
        if num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(self.true_labels, self.probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multi-class classification
            for i in range(num_classes):
                # One-vs-rest
                binary_labels = (self.true_labels == i).astype(int)
                fpr, tpr, _ = roc_curve(binary_labels, self.probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                plt.plot(fpr, tpr, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path=None, figsize=(10, 8)):
        """
        Plot Precision-Recall curve
        
        Args:
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        if self.probabilities is None:
            raise ValueError("Must run predict() first")
        
        num_classes = self.probabilities.shape[1]
        
        plt.figure(figsize=figsize)
        
        if num_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(
                self.true_labels, self.probabilities[:, 1]
            )
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label='Precision-Recall curve')
        else:
            # Multi-class classification
            for i in range(num_classes):
                binary_labels = (self.true_labels == i).astype(int)
                precision, recall, _ = precision_recall_curve(
                    binary_labels, self.probabilities[:, i]
                )
                
                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                plt.plot(recall, precision, lw=2, label=class_name)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history, save_path=None, figsize=(14, 5)):
        """
        Plot training history
        
        Args:
            history (dict): Training history
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def get_classification_report(self):
        """
        Get detailed classification report
        
        Returns:
            str: Classification report
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Must run predict() first")
        
        report = classification_report(
            self.true_labels, 
            self.predictions,
            target_names=self.class_names,
            digits=4
        )
        
        return report
    
    def print_classification_report(self):
        """
        Print classification report
        """
        report = self.get_classification_report()
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        print(report)
        print("=" * 50)
    
    def analyze_errors(self, texts=None, num_examples=10):
        """
        Analyze misclassified examples
        
        Args:
            texts (list): Original texts (optional)
            num_examples (int): Number of examples to show
            
        Returns:
            list: List of error examples
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Must run predict() first")
        
        # Find misclassified indices
        error_indices = np.where(self.predictions != self.true_labels)[0]
        
        print("\n" + "=" * 50)
        print(f"MISCLASSIFIED EXAMPLES ({len(error_indices)} total errors)")
        print("=" * 50)
        
        errors = []
        for i, idx in enumerate(error_indices[:num_examples]):
            true_label = self.true_labels[idx]
            pred_label = self.predictions[idx]
            
            true_name = self.class_names[true_label] if self.class_names else true_label
            pred_name = self.class_names[pred_label] if self.class_names else pred_label
            
            error_info = {
                'index': int(idx),
                'true_label': true_name,
                'predicted_label': pred_name,
                'confidence': float(self.probabilities[idx][pred_label])
            }
            
            if texts is not None:
                error_info['text'] = texts[idx]
            
            errors.append(error_info)
            
            print(f"\nExample {i+1}:")
            print(f"  True: {true_name}")
            print(f"  Predicted: {pred_name}")
            print(f"  Confidence: {error_info['confidence']:.4f}")
            if texts is not None:
                print(f"  Text: {texts[idx][:100]}...")
        
        return errors
    
    def save_results(self, save_dir):
        """
        Save all evaluation results
        
        Args:
            save_dir (str): Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics
        metrics = self.calculate_metrics()
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save classification report
        report = self.get_classification_report()
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Save plots
        self.plot_confusion_matrix(
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        self.plot_roc_curve(
            save_path=os.path.join(save_dir, 'roc_curve.png')
        )
        self.plot_precision_recall_curve(
            save_path=os.path.join(save_dir, 'precision_recall_curve.png')
        )
        
        print(f"\nAll results saved to {save_dir}")


# Example usage
if __name__ == "__main__":
    print("Evaluator Module")
    print("=" * 50)
    print("\nThis module provides comprehensive evaluation tools")
    print("for text classification models.")
