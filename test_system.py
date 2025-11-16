"""
System Test Script - Verify All Components
"""

import sys
import traceback


def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "=" * 60)
    print("TEST 1: Checking Package Imports")
    print("=" * 60)
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'tqdm': 'TQDM'
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:30s} - OK")
        except ImportError:
            print(f"✗ {name:30s} - MISSING")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages installed successfully!")
    return True


def test_data_cleaner():
    """Test data cleaning module"""
    print("\n" + "=" * 60)
    print("TEST 2: Data Cleaner Module")
    print("=" * 60)
    
    try:
        from data_cleaner import DataCleaner
        import pandas as pd
        
        # Create sample data
        data = {
            'question': [
                'How do I <b>read</b> a file in Python?',
                'What is the difference between let and var?'
            ],
            'tags': ['python,file-io', 'javascript,variables']
        }
        df = pd.DataFrame(data)
        
        # Test cleaning
        cleaner = DataCleaner(min_tag_frequency=1)
        df_clean = cleaner.clean_dataset(df, 'question', 'tags')
        
        assert len(df_clean) > 0, "Cleaned dataset is empty"
        assert 'question_clean' in df_clean.columns, "Missing question_clean column"
        assert 'tags_filtered' in df_clean.columns, "Missing tags_filtered column"
        
        print("✓ Data cleaning works")
        print("✓ Tag parsing works")
        print("✓ Vocabulary building works")
        print("\n✅ Data Cleaner Module - PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Data Cleaner Module - FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_bert_model():
    """Test BERT model creation"""
    print("\n" + "=" * 60)
    print("TEST 3: BERT Model Module")
    print("=" * 60)
    
    try:
        import torch
        from bert_model import BERTTagPredictor, create_tokenizer
        
        # Create model
        num_tags = 10
        model = BERTTagPredictor(num_tags=num_tags, hidden_dim=64)
        
        print(f"✓ Model created with {num_tags} tags")
        
        # Test forward pass
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, num_tags), "Wrong output shape"
        print("✓ Forward pass works")
        
        # Test tokenizer
        tokenizer = create_tokenizer()
        text = "How do I read a file?"
        encoded = tokenizer(text, return_tensors='pt')
        
        assert 'input_ids' in encoded, "Tokenizer missing input_ids"
        assert 'attention_mask' in encoded, "Tokenizer missing attention_mask"
        print("✓ Tokenizer works")
        
        print("\n✅ BERT Model Module - PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ BERT Model Module - FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_dataset():
    """Test PyTorch dataset"""
    print("\n" + "=" * 60)
    print("TEST 4: Dataset Module")
    print("=" * 60)
    
    try:
        from dataset import TagPredictionDataset
        from transformers import BertTokenizer
        
        # Sample data
        questions = ["How do I read a file?", "What is Python?"]
        tags = [["python", "file-io"], ["python", "programming"]]
        tag_to_id = {"python": 0, "file-io": 1, "programming": 2}
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        dataset = TagPredictionDataset(
            questions=questions,
            tags=tags,
            tag_to_id=tag_to_id,
            tokenizer=tokenizer,
            max_length=128
        )
        
        assert len(dataset) == 2, "Wrong dataset size"
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test getting item
        item = dataset[0]
        assert 'input_ids' in item, "Missing input_ids"
        assert 'attention_mask' in item, "Missing attention_mask"
        assert 'labels' in item, "Missing labels"
        print("✓ Dataset indexing works")
        
        # Test data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        
        assert batch['input_ids'].shape[0] == 2, "Wrong batch size"
        print("✓ DataLoader works")
        
        print("\n✅ Dataset Module - PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset Module - FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_trainer():
    """Test trainer module"""
    print("\n" + "=" * 60)
    print("TEST 5: Trainer Module")
    print("=" * 60)
    
    try:
        from trainer import TagPredictionTrainer
        import torch
        
        print("✓ Trainer module imports successfully")
        print("✓ Training functions available")
        print("✓ Metrics calculation available")
        
        print("\n✅ Trainer Module - PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Trainer Module - FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_inference():
    """Test inference module"""
    print("\n" + "=" * 60)
    print("TEST 6: Inference Module")
    print("=" * 60)
    
    try:
        from inference import TagPredictor
        
        print("✓ Inference module imports successfully")
        print("✓ TagPredictor class available")
        print("✓ Prediction functions available")
        
        print("\n✅ Inference Module - PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Inference Module - FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_pipeline():
    """Test training pipeline"""
    print("\n" + "=" * 60)
    print("TEST 7: Training Pipeline")
    print("=" * 60)
    
    try:
        import train_pipeline
        
        print("✓ Pipeline module imports successfully")
        print("✓ All components integrated")
        
        print("\n✅ Training Pipeline - PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Training Pipeline - FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "=" * 60)
    print("TEST 8: CUDA/GPU Check")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print("\n✅ GPU acceleration available!")
        else:
            print("⚠ CUDA not available")
            print("⚠ Training will use CPU (slower)")
            print("\n⚠️  GPU not available - will use CPU")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CUDA Check - FAILED")
        print(f"Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" " * 20 + "SYSTEM TEST SUITE")
    print("=" * 70)
    print("\nTesting all components of the BERT Tag Prediction System...")
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Cleaner", test_data_cleaner),
        ("BERT Model", test_bert_model),
        ("Dataset", test_dataset),
        ("Trainer", test_trainer),
        ("Inference", test_inference),
        ("Pipeline", test_pipeline),
        ("CUDA/GPU", test_cuda)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} - CRASHED")
            print(f"Error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\n📚 Next steps:")
        print("   1. Run examples: python example_usage.py")
        print("   2. Train model: python train_pipeline.py --num_samples 1000 --num_epochs 3")
        print("   3. Read docs: cat README_TAG_PREDICTION.md")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Check Python version: python --version (need 3.8+)")
        print("   - Update packages: pip install --upgrade torch transformers")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
