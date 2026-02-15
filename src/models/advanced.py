"""
Advanced Models for Text Classification
This module implements deep learning models:
- CNN for Text
- LSTM/BiLSTM
- BERT (Transformers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np


class TextCNN(nn.Module):
    """
    Convolutional Neural Network for Text Classification
    """
    
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, 
                 filter_sizes=[3, 4, 5], num_classes=2, dropout=0.5):
        """
        Initialize CNN model
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
            num_filters (int): Number of filters per filter size
            filter_sizes (list): List of filter sizes
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
        """
        super(TextCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim))
            for fs in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Output logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_length, embedding_dim]
        embedded = self.embedding(x)
        
        # Add channel dimension: [batch_size, 1, seq_length, embedding_dim]
        embedded = embedded.unsqueeze(1)
        
        # Apply convolution and pooling
        conv_outputs = []
        for conv in self.convs:
            # Convolution: [batch_size, num_filters, seq_length - filter_size + 1, 1]
            conv_out = F.relu(conv(embedded).squeeze(3))
            
            # Max pooling: [batch_size, num_filters]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate: [batch_size, num_filters * len(filter_sizes)]
        concatenated = torch.cat(conv_outputs, 1)
        
        # Dropout
        concatenated = self.dropout(concatenated)
        
        # Fully connected: [batch_size, num_classes]
        output = self.fc(concatenated)
        
        return output


class TextLSTM(nn.Module):
    """
    LSTM/BiLSTM for Text Classification
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, 
                 num_layers=2, num_classes=2, dropout=0.5, bidirectional=True):
        """
        Initialize LSTM model
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
            hidden_dim (int): Hidden dimension
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
            bidirectional (bool): Use bidirectional LSTM
        """
        super(TextLSTM, self).__init__()
        
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Output logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_length, embedding_dim]
        embedded = self.embedding(x)
        
        # LSTM: output [batch_size, seq_length, hidden_dim * num_directions]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # Fully connected: [batch_size, num_classes]
        output = self.fc(hidden)
        
        return output


class BERTClassifier:
    """
    BERT-based Text Classifier using Hugging Face Transformers
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, 
                 max_length=128, device=None):
        """
        Initialize BERT classifier
        
        Args:
            model_name (str): BERT model name
            num_classes (int): Number of output classes
            max_length (int): Maximum sequence length
            device: torch device
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        self.model.to(self.device)
    
    def encode_texts(self, texts):
        """
        Encode texts using BERT tokenizer
        
        Args:
            texts (list): List of text strings
            
        Returns:
            dict: Encoded inputs
        """
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoding
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids (torch.Tensor): Input token ids
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            Output from BERT model
        """
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        return outputs
    
    def predict(self, texts):
        """
        Make predictions on texts
        
        Args:
            texts (list): List of text strings
            
        Returns:
            np.array: Predicted class labels
        """
        self.model.eval()
        
        encoding = self.encode_texts(texts)
        
        with torch.no_grad():
            outputs = self.forward(
                encoding['input_ids'],
                encoding['attention_mask']
            )
            predictions = torch.argmax(outputs.logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def save_model(self, save_path):
        """
        Save model and tokenizer
        
        Args:
            save_path (str): Directory to save model
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        """
        Load model and tokenizer
        
        Args:
            load_path (str): Directory to load model from
        """
        self.model = BertForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")


class ModelFactory:
    """
    Factory class to create different model types
    """
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Create model based on type
        
        Args:
            model_type (str): Type of model ('cnn', 'lstm', 'bert')
            **kwargs: Model parameters
            
        Returns:
            Model instance
        """
        if model_type == 'cnn':
            return TextCNN(**kwargs)
        elif model_type == 'lstm':
            return TextLSTM(**kwargs)
        elif model_type == 'bert':
            return BERTClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    print("Advanced Models Module")
    print("=" * 50)
    
    # Example CNN
    print("\nCreating CNN model...")
    cnn_model = TextCNN(vocab_size=10000, num_classes=5)
    print(f"CNN parameters: {sum(p.numel() for p in cnn_model.parameters())}")
    
    # Example LSTM
    print("\nCreating LSTM model...")
    lstm_model = TextLSTM(vocab_size=10000, num_classes=5)
    print(f"LSTM parameters: {sum(p.numel() for p in lstm_model.parameters())}")
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_length = 20
    dummy_input = torch.randint(0, 10000, (batch_size, seq_length))
    
    print("\nTesting CNN forward pass...")
    cnn_output = cnn_model(dummy_input)
    print(f"CNN output shape: {cnn_output.shape}")
    
    print("\nTesting LSTM forward pass...")
    lstm_output = lstm_model(dummy_input)
    print(f"LSTM output shape: {lstm_output.shape}")
    
    print("\nAll models initialized successfully!")
