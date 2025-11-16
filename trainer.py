"""
Training Module for BERT Tag Prediction Model
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import json
import os


class TagPredictionTrainer:
    """Trainer for BERT tag prediction model"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: str = 'cuda',
                 learning_rate: float = 2e-5, num_epochs: int = 5,
                 warmup_steps: int = 0, weight_decay: float = 0.01,
                 save_dir: str = 'checkpoints'):
        """
        Initialize trainer
        
        Args:
            model: BERT tag prediction model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for optimizer
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function (Binary Cross Entropy for multi-label)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> float:
        """
        Train for one epoch
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model
        
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Get predictions (threshold at 0.5)
                predictions = torch.sigmoid(logits) > 0.5
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        metrics = self.calculate_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions: np.ndarray, 
                         labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            predictions: Binary predictions [batch_size, num_tags]
            labels: Ground truth labels [batch_size, num_tags]
            
        Returns:
            Dictionary of metrics
        """
        # Micro-averaged metrics
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions * (1 - labels))
        fn = np.sum((1 - predictions) * labels)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs
        
        Returns:
            Training history
        """
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, metrics = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_precision'].append(metrics['precision'])
            self.history['val_recall'].append(metrics['recall'])
            self.history['val_f1'].append(metrics['f1'])
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss, metrics)
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint every epoch
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, val_loss, metrics)
        
        # Save final model
        self.save_checkpoint('final_model.pt', self.num_epochs - 1, val_loss, metrics)
        
        # Save training history
        self.save_history()
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, 
                       val_loss: float, metrics: Dict[str, float]) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'history': self.history
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def save_history(self) -> None:
        """Save training history to JSON"""
        filepath = os.path.join(self.save_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {filepath}")


def evaluate_top_k(model: nn.Module, test_loader: DataLoader, 
                   id_to_tag: Dict[int, str], device: str = 'cuda',
                   k: int = 3) -> Dict[str, float]:
    """
    Evaluate top-k prediction accuracy
    
    Args:
        model: Trained model
        test_loader: Test data loader
        id_to_tag: ID to tag mapping
        device: Device
        k: Number of top predictions to consider
        
    Returns:
        Dictionary of top-k metrics
    """
    model.eval()
    
    correct_at_k = 0
    total_samples = 0
    total_tags = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating Top-K'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            
            # Get top-k predictions
            top_k_scores, top_k_indices = torch.topk(probs, k, dim=1)
            
            # Check if any true label is in top-k
            for i in range(len(labels)):
                true_labels = torch.where(labels[i] == 1)[0]
                predicted_labels = top_k_indices[i]
                
                # Check overlap
                overlap = len(set(true_labels.cpu().numpy()) & 
                            set(predicted_labels.cpu().numpy()))
                
                if overlap > 0:
                    correct_at_k += 1
                
                total_samples += 1
                total_tags += len(true_labels)
    
    accuracy_at_k = correct_at_k / total_samples
    
    return {
        f'accuracy@{k}': accuracy_at_k,
        'total_samples': total_samples,
        'avg_tags_per_sample': total_tags / total_samples
    }


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Use this module to train your BERT tag prediction model.")
