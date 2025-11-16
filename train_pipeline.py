"""
Complete Training Pipeline for BERT Tag Prediction
"""

import torch
import pandas as pd
import argparse
import os
from transformers import BertTokenizer

from data_cleaner import DataCleaner
from bert_model import BERTTagPredictor
from dataset import create_data_loaders
from trainer import TagPredictionTrainer, evaluate_top_k


def main(args):
    """
    Main training pipeline
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("BERT Tag Prediction Training Pipeline")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Step 1: Load and clean data
    print("\n" + "=" * 60)
    print("Step 1: Loading and Cleaning Data")
    print("=" * 60)
    
    if not os.path.exists(args.data_path):
        print(f"Data file not found: {args.data_path}")
        print("Creating sample dataset...")
        from data_cleaner import create_sample_dataset
        create_sample_dataset(args.data_path, num_samples=args.num_samples)
    
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples from {args.data_path}")
    
    # Clean data
    cleaner = DataCleaner(
        min_tag_frequency=args.min_tag_frequency,
        max_tags_per_question=args.max_tags_per_question
    )
    
    df_clean = cleaner.clean_dataset(
        df,
        question_column=args.question_column,
        tag_column=args.tag_column
    )
    
    # Save cleaned data
    cleaned_path = os.path.join(args.output_dir, 'cleaned_data.csv')
    df_clean.to_csv(cleaned_path, index=False)
    print(f"\nCleaned data saved to {cleaned_path}")
    
    # Save vocabulary
    vocab_path = os.path.join(args.output_dir, 'tag_vocabulary.json')
    cleaner.save_vocabulary(vocab_path)
    
    # Step 2: Create data loaders
    print("\n" + "=" * 60)
    print("Step 2: Creating Data Loaders")
    print("=" * 60)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        df=df_clean,
        tag_to_id=cleaner.tag_to_id,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    # Step 3: Create model
    print("\n" + "=" * 60)
    print("Step 3: Creating Model")
    print("=" * 60)
    
    model = BERTTagPredictor(
        num_tags=len(cleaner.tag_to_id),
        bert_model_name=args.bert_model,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {args.bert_model}")
    print(f"Number of tags: {len(cleaner.tag_to_id)}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 4: Train model
    print("\n" + "=" * 60)
    print("Step 4: Training Model")
    print("=" * 60)
    
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    
    trainer = TagPredictionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        save_dir=checkpoint_dir
    )
    
    history = trainer.train()
    
    # Step 5: Evaluate on test set
    print("\n" + "=" * 60)
    print("Step 5: Evaluating on Test Set")
    print("=" * 60)
    
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_top_k(
        model=model,
        test_loader=test_loader,
        id_to_tag=cleaner.id_to_tag,
        device=device,
        k=3
    )
    
    print("\nTest Set Results:")
    print("-" * 40)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Step 6: Save final artifacts
    print("\n" + "=" * 60)
    print("Step 6: Saving Artifacts")
    print("=" * 60)
    
    print(f"✓ Model checkpoints: {checkpoint_dir}")
    print(f"✓ Tag vocabulary: {vocab_path}")
    print(f"✓ Cleaned data: {cleaned_path}")
    print(f"✓ Training history: {os.path.join(checkpoint_dir, 'training_history.json')}")
    
    print("\n" + "=" * 60)
    print("Training Pipeline Completed Successfully!")
    print("=" * 60)
    
    print("\nTo use the trained model for inference:")
    print(f"python inference.py {best_model_path} {vocab_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT Tag Prediction Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='sample_data.csv',
                       help='Path to input data CSV file')
    parser.add_argument('--question_column', type=str, default='question',
                       help='Name of question column')
    parser.add_argument('--tag_column', type=str, default='tags',
                       help='Name of tag column')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for artifacts')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate if creating sample data')
    
    # Data cleaning arguments
    parser.add_argument('--min_tag_frequency', type=int, default=5,
                       help='Minimum tag frequency to include')
    parser.add_argument('--max_tags_per_question', type=int, default=5,
                       help='Maximum tags per question')
    
    # Model arguments
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                       help='Pre-trained BERT model name')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training data split ratio')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation data split ratio')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run pipeline
    main(args)
