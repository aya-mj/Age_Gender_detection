"""
BERT-based Multi-label Tag Prediction Model
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import List, Tuple, Dict
import numpy as np


class BERTTagPredictor(nn.Module):
    """BERT-based model for multi-label tag prediction"""
    
    def __init__(self, num_tags: int, bert_model_name: str = 'bert-base-uncased', 
                 dropout: float = 0.3, hidden_dim: int = 256):
        """
        Initialize BERT tag predictor
        
        Args:
            num_tags: Number of unique tags
            bert_model_name: Pre-trained BERT model name
            dropout: Dropout rate
            hidden_dim: Hidden layer dimension
        """
        super(BERTTagPredictor, self).__init__()
        
        self.num_tags = num_tags
        self.bert_model_name = bert_model_name
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Get BERT hidden size
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tags)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tag logits [batch_size, num_tags]
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict_top_k(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                     k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k tags
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            k: Number of top tags to return
            
        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            
            # Get top-k predictions
            top_k_scores, top_k_indices = torch.topk(probs, k, dim=1)
            
        return top_k_indices, top_k_scores


class BERTTagPredictorWithTitle(nn.Module):
    """Enhanced BERT model that processes both title and body"""
    
    def __init__(self, num_tags: int, bert_model_name: str = 'bert-base-uncased',
                 dropout: float = 0.3, hidden_dim: int = 256):
        """
        Initialize enhanced BERT tag predictor
        
        Args:
            num_tags: Number of unique tags
            bert_model_name: Pre-trained BERT model name
            dropout: Dropout rate
            hidden_dim: Hidden layer dimension
        """
        super(BERTTagPredictorWithTitle, self).__init__()
        
        self.num_tags = num_tags
        self.bert_model_name = bert_model_name
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Attention mechanism to combine title and body
        self.attention = nn.Sequential(
            nn.Linear(self.bert_hidden_size * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tags)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tag logits [batch_size, num_tags]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return logits


def create_tokenizer(model_name: str = 'bert-base-uncased') -> BertTokenizer:
    """
    Create BERT tokenizer
    
    Args:
        model_name: Pre-trained BERT model name
        
    Returns:
        BertTokenizer instance
    """
    return BertTokenizer.from_pretrained(model_name)


def tokenize_questions(questions: List[str], tokenizer: BertTokenizer, 
                      max_length: int = 128) -> Dict[str, torch.Tensor]:
    """
    Tokenize questions using BERT tokenizer
    
    Args:
        questions: List of question texts
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    encoded = tokenizer(
        questions,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing BERT Tag Predictor...")
    
    num_tags = 100
    model = BERTTagPredictor(num_tags=num_tags)
    
    print(f"Model created with {num_tags} tags")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    
    print("\nTesting forward pass...")
    logits = model(dummy_input_ids, dummy_attention_mask)
    print(f"Output shape: {logits.shape}")
    
    # Test top-k prediction
    print("\nTesting top-3 prediction...")
    top_k_indices, top_k_scores = model.predict_top_k(dummy_input_ids, dummy_attention_mask, k=3)
    print(f"Top-k indices shape: {top_k_indices.shape}")
    print(f"Top-k scores shape: {top_k_scores.shape}")
    
    print("\nModel test completed successfully!")
