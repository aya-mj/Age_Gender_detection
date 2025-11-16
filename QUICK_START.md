# Quick Start Guide - BERT Tag Prediction System

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install torch transformers pandas numpy scikit-learn tqdm datasets
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
# Train with sample data (for testing)
python train_pipeline.py --num_samples 1000 --num_epochs 3

# Or train with your own data
python train_pipeline.py --data_path your_data.csv --num_epochs 5
```

### Step 3: Make Predictions
```bash
python inference.py output/checkpoints/best_model.pt output/tag_vocabulary.json
```

Then enter questions like:
```
Question: How do I read a file in Python?

Top 3 Predicted Tags:
1. python               (confidence: 0.9234)
2. file-io              (confidence: 0.8567)
3. programming          (confidence: 0.7891)
```

---

## 📁 Complete File Structure

```
bert-tag-prediction/
├── requirements.txt              # Python dependencies
├── data_cleaner.py              # Data cleaning module
├── bert_model.py                # BERT model architecture
├── dataset.py                   # PyTorch dataset
├── trainer.py                   # Training logic
├── inference.py                 # Prediction module
├── train_pipeline.py            # Complete training pipeline
├── example_usage.py             # Usage examples
├── README_TAG_PREDICTION.md     # Full documentation
└── QUICK_START.md              # This file
```

---

## 📊 Your Data Format

Your CSV should look like this:

| question | tags |
|----------|------|
| How do I read a file in Python? | python,file-io,programming |
| What is REST API? | api,rest,web-services |
| How to center a div? | css,html,layout |

**Supported tag formats:**
- `python,file-io,programming` (comma-separated)
- `python|file-io|programming` (pipe-separated)
- `['python', 'file-io', 'programming']` (list format)

---

## 🎯 What This System Does

1. **Cleans your data**: Removes HTML, URLs, special characters
2. **Processes with BERT**: Uses state-of-the-art NLP model
3. **Trains classifier**: Multi-label classification for tags
4. **Predicts top 3 tags**: Returns most relevant tags with confidence scores

---

## 💡 Common Use Cases

### Use Case 1: Stack Overflow Style Q&A
```bash
python train_pipeline.py \
    --data_path stackoverflow_questions.csv \
    --question_column title \
    --tag_column tags \
    --num_epochs 10
```

### Use Case 2: Support Ticket Classification
```bash
python train_pipeline.py \
    --data_path support_tickets.csv \
    --question_column ticket_description \
    --tag_column categories \
    --min_tag_frequency 20
```

### Use Case 3: Document Categorization
```bash
python train_pipeline.py \
    --data_path documents.csv \
    --question_column document_text \
    --tag_column topics \
    --max_length 256
```

---

## 🔧 Key Parameters Explained

| Parameter | What it does | Recommended |
|-----------|--------------|-------------|
| `--num_epochs` | Training iterations | 5-10 for real data |
| `--batch_size` | Samples per batch | 16-32 (depends on GPU) |
| `--learning_rate` | How fast model learns | 2e-5 (default is good) |
| `--min_tag_frequency` | Min tag occurrences | 10-50 for large datasets |
| `--max_length` | Max question length | 128 (increase for long text) |

---

## 📈 Expected Results

After training, you'll get:

```
output/
├── cleaned_data.csv              # Your cleaned dataset
├── tag_vocabulary.json           # Tag mappings (ID ↔ Tag)
└── checkpoints/
    ├── best_model.pt            # Best performing model
    ├── final_model.pt           # Final model
    └── training_history.json    # Training metrics
```

**Typical Performance:**
- Training time: 5-30 minutes (depends on data size and GPU)
- Accuracy@3: 70-90% (at least one correct tag in top 3)
- F1 Score: 0.6-0.8 (depends on data quality)

---

## 🐛 Troubleshooting

### Problem: Out of Memory
**Solution:** Reduce batch size
```bash
python train_pipeline.py --batch_size 8
```

### Problem: CUDA not available
**Solution:** It will automatically use CPU (slower but works)

### Problem: Poor predictions
**Solutions:**
- Increase training data (>1000 samples recommended)
- Increase epochs: `--num_epochs 10`
- Clean your data better
- Increase `--min_tag_frequency` to focus on common tags

### Problem: Training too slow
**Solutions:**
- Use GPU if available
- Reduce `--max_length 64`
- Use smaller BERT: `--bert_model distilbert-base-uncased`

---

## 🎓 Learning Path

1. **Start here**: Run `python example_usage.py` to see all examples
2. **Test with sample data**: `python train_pipeline.py --num_samples 500 --num_epochs 2`
3. **Use your data**: Prepare your CSV and train
4. **Optimize**: Adjust parameters for better results
5. **Deploy**: Use `inference.py` for predictions

---

## 📚 Module Overview

### 1. `data_cleaner.py`
- Cleans question text
- Parses and normalizes tags
- Builds tag vocabulary
- Filters by frequency

### 2. `bert_model.py`
- BERT-based neural network
- Multi-label classification head
- Top-K prediction support

### 3. `dataset.py`
- PyTorch dataset wrapper
- Tokenization
- Multi-hot label encoding

### 4. `trainer.py`
- Training loop
- Validation
- Metrics calculation
- Checkpoint saving

### 5. `inference.py`
- Load trained model
- Predict tags for new questions
- Batch prediction support
- Interactive mode

### 6. `train_pipeline.py`
- End-to-end pipeline
- Combines all modules
- Command-line interface

---

## 🔥 Pro Tips

1. **Start small**: Test with 500-1000 samples first
2. **Monitor training**: Watch validation loss - should decrease
3. **Tag frequency matters**: Remove rare tags with `--min_tag_frequency`
4. **GPU speeds up 10x**: Use CUDA if available
5. **Longer text**: Increase `--max_length` for detailed questions
6. **Better model**: Use `bert-large-uncased` for higher accuracy

---

## 📞 Need Help?

1. Check `README_TAG_PREDICTION.md` for detailed documentation
2. Run `python example_usage.py` to see working examples
3. Review code comments in each module
4. Test individual modules (each has `if __name__ == "__main__"` section)

---

## ✅ Checklist

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Prepared data in CSV format
- [ ] Ran example script (`python example_usage.py`)
- [ ] Trained model (`python train_pipeline.py`)
- [ ] Tested predictions (`python inference.py`)
- [ ] Read full documentation (`README_TAG_PREDICTION.md`)

---

## 🎉 You're Ready!

Now you have a complete BERT-based tag prediction system. Start with the examples and customize for your needs!

**Happy Tagging! 🏷️**
