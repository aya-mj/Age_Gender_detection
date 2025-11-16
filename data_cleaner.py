"""
Data Cleaning and Preprocessing Module for Tag Prediction
Handles cleaning of question-tag datasets
"""

import pandas as pd
import re
import numpy as np
from typing import List, Tuple
import json


class DataCleaner:
    """Clean and preprocess question-tag dataset"""
    
    def __init__(self, min_tag_frequency: int = 10, max_tags_per_question: int = 5):
        """
        Initialize data cleaner
        
        Args:
            min_tag_frequency: Minimum number of occurrences for a tag to be included
            max_tags_per_question: Maximum number of tags to keep per question
        """
        self.min_tag_frequency = min_tag_frequency
        self.max_tags_per_question = max_tags_per_question
        self.tag_to_id = {}
        self.id_to_tag = {}
        
    def clean_text(self, text: str) -> str:
        """
        Clean question text
        
        Args:
            text: Raw question text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        
        # Remove code blocks (common in Stack Overflow)
        text = re.sub(r'```[\s\S]*?```', ' ', text)
        text = re.sub(r'`[^`]*`', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def parse_tags(self, tags: str) -> List[str]:
        """
        Parse tags from various formats
        
        Args:
            tags: Tags in string format (comma-separated, pipe-separated, or list-like)
            
        Returns:
            List of cleaned tags
        """
        if pd.isna(tags) or not isinstance(tags, str):
            return []
        
        # Handle different tag formats
        if '|' in tags:
            tag_list = tags.split('|')
        elif ',' in tags:
            tag_list = tags.split(',')
        elif tags.startswith('[') and tags.endswith(']'):
            # Handle list-like strings
            try:
                tag_list = json.loads(tags.replace("'", '"'))
            except:
                tag_list = tags.strip('[]').replace("'", "").split(',')
        else:
            tag_list = [tags]
        
        # Clean individual tags
        cleaned_tags = []
        for tag in tag_list:
            tag = tag.strip().lower()
            tag = re.sub(r'[^a-z0-9\-\+\#]', '', tag)
            if tag and len(tag) > 1:
                cleaned_tags.append(tag)
        
        return cleaned_tags[:self.max_tags_per_question]
    
    def build_tag_vocabulary(self, df: pd.DataFrame, tag_column: str = 'tags') -> None:
        """
        Build tag vocabulary based on frequency
        
        Args:
            df: DataFrame with tags
            tag_column: Name of the column containing tags
        """
        tag_counts = {}
        
        for tags in df[tag_column]:
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Filter tags by minimum frequency
        valid_tags = [tag for tag, count in tag_counts.items() 
                     if count >= self.min_tag_frequency]
        
        # Sort tags by frequency (most common first)
        valid_tags = sorted(valid_tags, 
                          key=lambda x: tag_counts[x], 
                          reverse=True)
        
        # Create mappings
        self.tag_to_id = {tag: idx for idx, tag in enumerate(valid_tags)}
        self.id_to_tag = {idx: tag for tag, idx in self.tag_to_id.items()}
        
        print(f"Built vocabulary with {len(self.tag_to_id)} tags")
        print(f"Top 10 tags: {valid_tags[:10]}")
    
    def clean_dataset(self, df: pd.DataFrame, 
                     question_column: str = 'question',
                     tag_column: str = 'tags') -> pd.DataFrame:
        """
        Clean entire dataset
        
        Args:
            df: Raw DataFrame
            question_column: Name of question column
            tag_column: Name of tag column
            
        Returns:
            Cleaned DataFrame
        """
        print(f"Original dataset size: {len(df)}")
        
        # Create a copy
        df_clean = df.copy()
        
        # Clean questions
        print("Cleaning questions...")
        df_clean['question_clean'] = df_clean[question_column].apply(self.clean_text)
        
        # Parse tags
        print("Parsing tags...")
        df_clean['tags_parsed'] = df_clean[tag_column].apply(self.parse_tags)
        
        # Remove rows with empty questions or no tags
        df_clean = df_clean[
            (df_clean['question_clean'].str.len() > 10) & 
            (df_clean['tags_parsed'].apply(len) > 0)
        ]
        
        print(f"After removing empty questions/tags: {len(df_clean)}")
        
        # Build tag vocabulary
        self.build_tag_vocabulary(df_clean, 'tags_parsed')
        
        # Filter tags to only include those in vocabulary
        df_clean['tags_filtered'] = df_clean['tags_parsed'].apply(
            lambda tags: [tag for tag in tags if tag in self.tag_to_id]
        )
        
        # Remove rows with no valid tags
        df_clean = df_clean[df_clean['tags_filtered'].apply(len) > 0]
        
        print(f"Final dataset size: {len(df_clean)}")
        
        return df_clean
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save tag vocabulary to file"""
        vocab_data = {
            'tag_to_id': self.tag_to_id,
            'id_to_tag': self.id_to_tag,
            'num_tags': len(self.tag_to_id)
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load tag vocabulary from file"""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        self.tag_to_id = vocab_data['tag_to_id']
        self.id_to_tag = {int(k): v for k, v in vocab_data['id_to_tag'].items()}
        print(f"Loaded vocabulary with {vocab_data['num_tags']} tags")


def create_sample_dataset(output_path: str = 'sample_data.csv', num_samples: int = 1000):
    """
    Create a sample dataset for demonstration
    
    Args:
        output_path: Path to save the sample dataset
        num_samples: Number of samples to generate
    """
    questions = [
        "How do I read a file in Python?",
        "What is the difference between let and var in JavaScript?",
        "How to center a div in CSS?",
        "How do I connect to a MySQL database?",
        "What is a REST API?",
        "How to sort an array in Java?",
        "What is the difference between SQL and NoSQL?",
        "How do I use async/await in JavaScript?",
        "What is object-oriented programming?",
        "How to create a virtual environment in Python?",
    ]
    
    tags = [
        "python,file-io,programming",
        "javascript,variables,scope",
        "css,html,layout",
        "mysql,database,connection",
        "api,rest,web-services",
        "java,arrays,sorting",
        "database,sql,nosql",
        "javascript,async,promises",
        "oop,programming,concepts",
        "python,virtualenv,environment",
    ]
    
    # Generate more samples by combining and modifying
    import random
    
    data = []
    for i in range(num_samples):
        q_idx = i % len(questions)
        t_idx = i % len(tags)
        data.append({
            'id': i,
            'question': questions[q_idx] + f" (variant {i})",
            'tags': tags[t_idx]
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created: {output_path}")
    return df


if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Clean the dataset
    cleaner = DataCleaner(min_tag_frequency=5, max_tags_per_question=5)
    df_clean = cleaner.clean_dataset(df, question_column='question', tag_column='tags')
    
    # Save cleaned data
    df_clean.to_csv('cleaned_data.csv', index=False)
    print("\nCleaned data saved to cleaned_data.csv")
    
    # Save vocabulary
    cleaner.save_vocabulary('tag_vocabulary.json')
    
    print("\nSample cleaned data:")
    print(df_clean[['question_clean', 'tags_filtered']].head())
