"""
PyTorch Dataset for Tag Prediction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from transformers import BertTokenizer


class TagPredictionDataset(Dataset):
    """PyTorch Dataset for tag prediction"""
    
    def __init__(self, questions: List[str], tags: List[List[str]], 
                 tag_to_id: Dict[str, int], tokenizer: BertTokenizer,
                 max_length: int = 128):
        """
        Initialize dataset
        
        Args:
            questions: List of question texts
            tags: List of tag lists for each question
            tag_to_id: Mapping from tag name to tag ID
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.questions = questions
        self.tags = tags
        self.tag_to_id = tag_to_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_tags = len(tag_to_id)
        
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        question = self.questions[idx]
        tags = self.tags[idx]
        
        # Tokenize question
        encoded = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create multi-hot encoded labels
        labels = torch.zeros(self.num_tags, dtype=torch.float32)
        for tag in tags:
            if tag in self.tag_to_id:
                labels[self.tag_to_id[tag]] = 1.0
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels
        }


def create_data_loaders(df: pd.DataFrame, tag_to_id: Dict[str, int],
                       tokenizer: BertTokenizer, batch_size: int = 16,
                       max_length: int = 128, train_split: float = 0.8,
                       val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        df: DataFrame with 'question_clean' and 'tags_filtered' columns
        tag_to_id: Tag to ID mapping
        tokenizer: BERT tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        train_split: Training data proportion
        val_split: Validation data proportion
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Create datasets
    train_dataset = TagPredictionDataset(
        questions=train_df['question_clean'].tolist(),
        tags=train_df['tags_filtered'].tolist(),
        tag_to_id=tag_to_id,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = TagPredictionDataset(
        questions=val_df['question_clean'].tolist(),
        tags=val_df['tags_filtered'].tolist(),
        tag_to_id=tag_to_id,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = TagPredictionDataset(
        questions=test_df['question_clean'].tolist(),
        tags=test_df['tags_filtered'].tolist(),
        tag_to_id=tag_to_id,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset creation
    print("Testing TagPredictionDataset...")
    
    from transformers import BertTokenizer
    
    # Sample data
    questions = [
        "How do I read a file in Python?",
        "What is the difference between let and var in JavaScript?",
        "How to center a div in CSS?"
    ]
    
    tags = [
        ["python", "file-io"],
        ["javascript", "variables"],
        ["css", "html"]
    ]
    
    tag_to_id = {
        "python": 0,
        "file-io": 1,
        "javascript": 2,
        "variables": 3,
        "css": 4,
        "html": 5
    }
    
    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset
    dataset = TagPredictionDataset(
        questions=questions,
        tags=tags,
        tag_to_id=tag_to_id,
        tokenizer=tokenizer,
        max_length=128
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test getting an item
    item = dataset[0]
    print(f"\nSample item:")
    print(f"Input IDs shape: {item['input_ids'].shape}")
    print(f"Attention mask shape: {item['attention_mask'].shape}")
    print(f"Labels shape: {item['labels'].shape}")
    print(f"Labels: {item['labels']}")
    
    # Test data loader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    
    print(f"\nSample batch:")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    print("\nDataset test completed successfully!")
