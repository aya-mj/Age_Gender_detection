"""
Example Usage Script - Complete Workflow Demonstration
"""

import os
import pandas as pd
from data_cleaner import DataCleaner, create_sample_dataset


def example_1_data_cleaning():
    """Example 1: Data cleaning and preprocessing"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Data Cleaning and Preprocessing")
    print("=" * 60)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    df = create_sample_dataset('example_data.csv', num_samples=100)
    print(f"   Created {len(df)} samples")
    
    # Initialize cleaner
    print("\n2. Initializing data cleaner...")
    cleaner = DataCleaner(min_tag_frequency=3, max_tags_per_question=5)
    
    # Clean dataset
    print("\n3. Cleaning dataset...")
    df_clean = cleaner.clean_dataset(df, question_column='question', tag_column='tags')
    
    # Display results
    print("\n4. Sample cleaned data:")
    print("-" * 60)
    for idx, row in df_clean.head(3).iterrows():
        print(f"\nQuestion: {row['question_clean'][:80]}...")
        print(f"Tags: {row['tags_filtered']}")
    
    # Save vocabulary
    cleaner.save_vocabulary('example_vocab.json')
    
    return df_clean, cleaner


def example_2_model_creation():
    """Example 2: Creating and testing the model"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Model Creation and Testing")
    print("=" * 60)
    
    from bert_model import BERTTagPredictor, create_tokenizer
    import torch
    
    # Create model
    print("\n1. Creating BERT model...")
    num_tags = 20
    model = BERTTagPredictor(num_tags=num_tags, hidden_dim=128)
    
    print(f"   Model created with {num_tags} tags")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create tokenizer
    print("\n2. Creating tokenizer...")
    tokenizer = create_tokenizer()
    
    # Test with sample question
    print("\n3. Testing with sample question...")
    question = "How do I read a file in Python?"
    
    encoded = tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    print(f"   Question: {question}")
    print(f"   Tokenized shape: {encoded['input_ids'].shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(encoded['input_ids'], encoded['attention_mask'])
    
    print(f"   Output shape: {logits.shape}")
    print("   ✓ Model working correctly!")
    
    return model, tokenizer


def example_3_dataset_creation():
    """Example 3: Creating PyTorch dataset"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: PyTorch Dataset Creation")
    print("=" * 60)
    
    from dataset import TagPredictionDataset
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
        "python": 0, "file-io": 1, "javascript": 2,
        "variables": 3, "css": 4, "html": 5
    }
    
    # Create dataset
    print("\n1. Creating dataset...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = TagPredictionDataset(
        questions=questions,
        tags=tags,
        tag_to_id=tag_to_id,
        tokenizer=tokenizer,
        max_length=128
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Get sample
    print("\n2. Sample item from dataset:")
    item = dataset[0]
    print(f"   Input IDs shape: {item['input_ids'].shape}")
    print(f"   Attention mask shape: {item['attention_mask'].shape}")
    print(f"   Labels shape: {item['labels'].shape}")
    print(f"   Active labels: {item['labels'].nonzero().squeeze().tolist()}")
    
    return dataset


def example_4_training_workflow():
    """Example 4: Complete training workflow"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Complete Training Workflow")
    print("=" * 60)
    
    print("\nTo run the complete training pipeline:")
    print("-" * 60)
    
    print("\n1. Basic training with sample data:")
    print("   python train_pipeline.py --num_samples 1000 --num_epochs 3")
    
    print("\n2. Training with your own data:")
    print("   python train_pipeline.py \\")
    print("       --data_path your_data.csv \\")
    print("       --question_column question \\")
    print("       --tag_column tags \\")
    print("       --num_epochs 5 \\")
    print("       --batch_size 16")
    
    print("\n3. Advanced training:")
    print("   python train_pipeline.py \\")
    print("       --data_path data.csv \\")
    print("       --bert_model bert-large-uncased \\")
    print("       --batch_size 32 \\")
    print("       --num_epochs 10 \\")
    print("       --learning_rate 2e-5 \\")
    print("       --max_length 256 \\")
    print("       --output_dir my_model")


def example_5_inference():
    """Example 5: Inference workflow"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Inference Workflow")
    print("=" * 60)
    
    print("\nAfter training, use the model for predictions:")
    print("-" * 60)
    
    print("\n1. Interactive mode:")
    print("   python inference.py \\")
    print("       output/checkpoints/best_model.pt \\")
    print("       output/tag_vocabulary.json")
    
    print("\n2. Programmatic usage:")
    print("""
from inference import TagPredictor

# Load model
predictor = TagPredictor(
    model_path='output/checkpoints/best_model.pt',
    vocab_path='output/tag_vocabulary.json'
)

# Predict
question = "How do I sort an array in Python?"
predictions = predictor.predict(question, top_k=3)

for tag, score in predictions:
    print(f"{tag}: {score:.4f}")
""")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print(" " * 15 + "BERT TAG PREDICTION - EXAMPLES")
    print("=" * 70)
    
    try:
        # Example 1: Data cleaning
        df_clean, cleaner = example_1_data_cleaning()
        
        # Example 2: Model creation
        model, tokenizer = example_2_model_creation()
        
        # Example 3: Dataset creation
        dataset = example_3_dataset_creation()
        
        # Example 4: Training workflow
        example_4_training_workflow()
        
        # Example 5: Inference workflow
        example_5_inference()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
        print("\n📚 Next Steps:")
        print("-" * 70)
        print("1. Review the generated files:")
        print("   - example_data.csv (sample dataset)")
        print("   - example_vocab.json (tag vocabulary)")
        
        print("\n2. Run the complete training pipeline:")
        print("   python train_pipeline.py --num_samples 1000 --num_epochs 3")
        
        print("\n3. After training, test inference:")
        print("   python inference.py output/checkpoints/best_model.pt output/tag_vocabulary.json")
        
        print("\n4. Read the documentation:")
        print("   cat README_TAG_PREDICTION.md")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
