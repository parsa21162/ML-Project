"""
Training Module for Deep Learning Models
Handles training loop, validation, checkpointing, and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime


class TextDataset(Dataset):
    """
    Custom Dataset for text classification
    """
    
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        """
        Initialize dataset
        
        Args:
            texts (list): List of text strings or token indices
            labels (list): List of labels
            tokenizer: Tokenizer for BERT models (optional)
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer is not None:
            # For BERT models
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # For CNN/LSTM models
            # Assuming text is already converted to indices
            if isinstance(text, list):
                # Pad or truncate
                if len(text) < self.max_length:
                    text = text + [0] * (self.max_length - len(text))
                else:
                    text = text[:self.max_length]
                text = torch.tensor(text, dtype=torch.long)
            
            return {
                'input': text,
                'label': torch.tensor(label, dtype=torch.long)
            }


class Trainer:
    """
    Trainer class for deep learning models
    """
    
    def __init__(self, model, device=None, learning_rate=0.001, 
                 weight_decay=0.0001, model_type='cnn'):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: torch device
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
            model_type (str): Type of model ('cnn', 'lstm', 'bert')
        """
        self.model = model
        self.model_type = model_type
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        if model_type != 'bert':  # BERT handles device internally
            self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if model_type == 'bert':
            # Use AdamW for BERT
            self.optimizer = optim.AdamW(
                self.model.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move data to device
            if self.model_type == 'bert':
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits
            else:
                inputs = batch['input'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move data to device
                if self.model_type == 'bert':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                else:
                    inputs = batch['input'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Forward pass
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=10, 
              patience=3, save_path='./models', save_best=True):
        """
        Complete training loop
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            num_epochs (int): Number of epochs
            patience (int): Early stopping patience
            save_path (str): Path to save models
            save_best (bool): Save best model
        """
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Early stopping patience: {patience}")
        print("=" * 50)
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print statistics
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save model
                model_path = os.path.join(save_path, 'best_model.pt')
                if self.model_type == 'bert':
                    self.model.save_model(os.path.join(save_path, 'best_bert_model'))
                else:
                    torch.save(self.model.state_dict(), model_path)
                print(f"✓ Best model saved! (Val Acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{patience}")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print("\n" + "=" * 50)
        print("Training complete!")
        print(f"Best Val Acc: {self.best_val_acc:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        
        # Save training history
        history_path = os.path.join(save_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {history_path}")
    
    def save_checkpoint(self, epoch, save_path):
        """
        Save training checkpoint
        
        Args:
            epoch (int): Current epoch
            save_path (str): Path to save checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load training checkpoint
        
        Args:
            checkpoint_path (str): Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']


# Example usage
if __name__ == "__main__":
    print("Trainer Module")
    print("=" * 50)
    
    # Example: Create dummy data
    from ..models.advanced import TextCNN
    
    print("\nCreating dummy dataset...")
    dummy_texts = [torch.randint(0, 1000, (50,)) for _ in range(100)]
    dummy_labels = [np.random.randint(0, 2) for _ in range(100)]
    
    dataset = TextDataset(dummy_texts, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("\nCreating model...")
    model = TextCNN(vocab_size=1000, num_classes=2)
    
    print("\nInitializing trainer...")
    trainer = Trainer(model, learning_rate=0.001)
    
    print("\nTrainer initialized successfully!")
