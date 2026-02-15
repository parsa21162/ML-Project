"""
Utility Helper Functions
Various helper functions for the project
"""

import numpy as np
import random
import torch
import os
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def count_parameters(model):
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total


def save_config(config, save_path):
    """
    Save configuration to JSON file
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save file
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {save_path}")


def load_config(config_path):
    """
    Load configuration from JSON file
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {config_path}")
    return config


def plot_word_cloud(texts, title="Word Cloud", max_words=100, 
                    figsize=(12, 8), save_path=None):
    """
    Generate and plot word cloud
    
    Args:
        texts (list or str): Text data
        title (str): Plot title
        max_words (int): Maximum number of words
        figsize (tuple): Figure size
        save_path (str): Path to save plot
    """
    # Combine texts if list
    if isinstance(texts, list):
        text = ' '.join(texts)
    else:
        text = texts
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color='white',
        colormap='viridis'
    ).generate(text)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Word cloud saved to {save_path}")
    
    plt.show()


def plot_text_length_distribution(texts, labels=None, max_length=None,
                                   figsize=(12, 6), save_path=None):
    """
    Plot distribution of text lengths
    
    Args:
        texts (list): List of texts
        labels (list): List of labels (optional)
        max_length (int): Maximum length to plot
        figsize (tuple): Figure size
        save_path (str): Path to save plot
    """
    lengths = [len(text.split()) for text in texts]
    
    if max_length:
        lengths = [min(l, max_length) for l in lengths]
    
    plt.figure(figsize=figsize)
    
    if labels is not None:
        # Plot by class
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            class_lengths = [lengths[i] for i in range(len(lengths)) if labels[i] == label]
            plt.hist(class_lengths, bins=50, alpha=0.5, label=f'Class {label}')
        plt.legend()
    else:
        plt.hist(lengths, bins=50, alpha=0.7, color='blue')
    
    plt.xlabel('Text Length (words)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Text Lengths', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Text length distribution saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\nText Length Statistics:")
    print(f"Mean: {np.mean(lengths):.2f} words")
    print(f"Median: {np.median(lengths):.2f} words")
    print(f"Min: {np.min(lengths)} words")
    print(f"Max: {np.max(lengths)} words")
    print(f"Std: {np.std(lengths):.2f} words")


def plot_class_distribution(labels, class_names=None, figsize=(10, 6), save_path=None):
    """
    Plot class distribution
    
    Args:
        labels (list or np.array): Labels
        class_names (list): Class names (optional)
        figsize (tuple): Figure size
        save_path (str): Path to save plot
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(unique)), counts, alpha=0.7, color='steelblue')
    
    # Color bars
    colors = sns.color_palette('husl', len(unique))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Labels
    if class_names:
        plt.xticks(range(len(unique)), class_names, rotation=45, ha='right')
    else:
        plt.xticks(range(len(unique)), [f'Class {i}' for i in unique])
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\nClass Distribution:")
    for i, (label, count) in enumerate(zip(unique, counts)):
        class_name = class_names[i] if class_names else f"Class {label}"
        print(f"{class_name}: {count} ({count/len(labels)*100:.2f}%)")


def create_vocab_from_texts(texts, max_vocab_size=None, min_freq=1):
    """
    Create vocabulary from texts
    
    Args:
        texts (list): List of tokenized texts
        max_vocab_size (int): Maximum vocabulary size
        min_freq (int): Minimum frequency for a word
        
    Returns:
        tuple: (word2idx, idx2word)
    """
    # Count word frequencies
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Filter by frequency
    word_freq = {w: f for w, f in word_freq.items() if f >= min_freq}
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Limit vocabulary size
    if max_vocab_size:
        sorted_words = sorted_words[:max_vocab_size-2]  # Reserve space for PAD and UNK
    
    # Create mappings
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, _) in enumerate(sorted_words, start=2):
        word2idx[word] = i
    
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"PAD token: {word2idx['<PAD>']}")
    print(f"UNK token: {word2idx['<UNK>']}")
    
    return word2idx, idx2word


def text_to_indices(text, word2idx, max_length=None):
    """
    Convert text to indices using vocabulary
    
    Args:
        text (str): Input text
        word2idx (dict): Word to index mapping
        max_length (int): Maximum length
        
    Returns:
        list: List of indices
    """
    indices = []
    for word in text.split():
        idx = word2idx.get(word, word2idx['<UNK>'])
        indices.append(idx)
    
    # Pad or truncate
    if max_length:
        if len(indices) < max_length:
            indices += [word2idx['<PAD>']] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
    
    return indices


def print_model_summary(model, input_size=None):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_size (tuple): Input size for test
    """
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(model)
    print("=" * 70)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 70)
    
    # Test forward pass if input_size provided
    if input_size:
        try:
            device = next(model.parameters()).device
            dummy_input = torch.randint(0, 1000, input_size).to(device)
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            print("=" * 70)
        except Exception as e:
            print(f"Could not perform test forward pass: {e}")


def calculate_metrics_from_confusion_matrix(cm):
    """
    Calculate metrics from confusion matrix
    
    Args:
        cm (np.array): Confusion matrix
        
    Returns:
        dict: Dictionary of metrics
    """
    # For binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    else:
        # For multi-class
        accuracy = np.trace(cm) / np.sum(cm)
        return {'accuracy': accuracy}


# Example usage
if __name__ == "__main__":
    print("Utility Module")
    print("=" * 50)
    
    # Set seed
    set_seed(42)
    
    # Example texts
    sample_texts = [
        "This is a great product",
        "Terrible service",
        "Amazing quality",
        "Not recommended"
    ]
    
    # Plot word cloud
    plot_word_cloud(sample_texts, title="Sample Word Cloud")
    
    print("\nUtility functions loaded successfully!")
