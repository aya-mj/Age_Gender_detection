"""
Inference Module for Tag Prediction
"""

import torch
import json
from typing import List, Tuple, Dict
from transformers import BertTokenizer
from bert_model import BERTTagPredictor
import numpy as np


class TagPredictor:
    """Inference class for tag prediction"""
    
    def __init__(self, model_path: str, vocab_path: str, 
                 bert_model_name: str = 'bert-base-uncased',
                 device: str = 'cuda'):
        """
        Initialize tag predictor
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to tag vocabulary JSON
            bert_model_name: BERT model name
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.tag_to_id = vocab_data['tag_to_id']
        self.id_to_tag = {int(k): v for k, v in vocab_data['id_to_tag'].items()}
        self.num_tags = vocab_data['num_tags']
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Load model
        self.model = BERTTagPredictor(
            num_tags=self.num_tags,
            bert_model_name=bert_model_name
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Vocabulary size: {self.num_tags}")
        print(f"Device: {self.device}")
    
    def predict(self, question: str, top_k: int = 3, 
                return_scores: bool = True) -> List[Tuple[str, float]]:
        """
        Predict top-k tags for a question
        
        Args:
            question: Question text
            top_k: Number of top tags to return
            return_scores: Whether to return confidence scores
            
        Returns:
            List of (tag, score) tuples if return_scores=True, else list of tags
        """
        # Tokenize
        encoded = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
        
        # Get top-k
        top_k_scores, top_k_indices = torch.topk(probs[0], top_k)
        
        # Convert to tags
        results = []
        for idx, score in zip(top_k_indices.cpu().numpy(), 
                             top_k_scores.cpu().numpy()):
            tag = self.id_to_tag[idx]
            if return_scores:
                results.append((tag, float(score)))
            else:
                results.append(tag)
        
        return results
    
    def predict_batch(self, questions: List[str], top_k: int = 3,
                     batch_size: int = 32) -> List[List[Tuple[str, float]]]:
        """
        Predict tags for multiple questions
        
        Args:
            questions: List of question texts
            top_k: Number of top tags to return per question
            batch_size: Batch size for processing
            
        Returns:
            List of predictions for each question
        """
        all_results = []
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_questions,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits)
            
            # Get top-k for each question
            for j in range(len(batch_questions)):
                top_k_scores, top_k_indices = torch.topk(probs[j], top_k)
                
                results = []
                for idx, score in zip(top_k_indices.cpu().numpy(),
                                     top_k_scores.cpu().numpy()):
                    tag = self.id_to_tag[idx]
                    results.append((tag, float(score)))
                
                all_results.append(results)
        
        return all_results
    
    def predict_with_threshold(self, question: str, 
                              threshold: float = 0.5,
                              max_tags: int = 10) -> List[Tuple[str, float]]:
        """
        Predict tags above a confidence threshold
        
        Args:
            question: Question text
            threshold: Minimum confidence threshold
            max_tags: Maximum number of tags to return
            
        Returns:
            List of (tag, score) tuples above threshold
        """
        # Tokenize
        encoded = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
        
        # Get all predictions above threshold
        probs_np = probs[0].cpu().numpy()
        above_threshold = np.where(probs_np >= threshold)[0]
        
        # Sort by score
        sorted_indices = above_threshold[np.argsort(-probs_np[above_threshold])]
        
        # Limit to max_tags
        sorted_indices = sorted_indices[:max_tags]
        
        results = []
        for idx in sorted_indices:
            tag = self.id_to_tag[idx]
            score = float(probs_np[idx])
            results.append((tag, score))
        
        return results


def interactive_prediction(predictor: TagPredictor):
    """
    Interactive mode for tag prediction
    
    Args:
        predictor: TagPredictor instance
    """
    print("\n" + "=" * 60)
    print("Interactive Tag Prediction")
    print("=" * 60)
    print("Enter questions to get tag predictions.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        # Predict
        predictions = predictor.predict(question, top_k=3, return_scores=True)
        
        print("\nTop 3 Predicted Tags:")
        print("-" * 40)
        for i, (tag, score) in enumerate(predictions, 1):
            print(f"{i}. {tag:20s} (confidence: {score:.4f})")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <vocab_path>")
        print("\nExample:")
        print("python inference.py checkpoints/best_model.pt tag_vocabulary.json")
        sys.exit(1)
    
    model_path = sys.argv[1]
    vocab_path = sys.argv[2]
    
    # Create predictor
    predictor = TagPredictor(
        model_path=model_path,
        vocab_path=vocab_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run interactive mode
    interactive_prediction(predictor)
