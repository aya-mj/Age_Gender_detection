# BERT-based Tag Prediction System

A complete implementation of a BERT-based multi-label classification system that predicts the top 3 tags for questions using NLP.

## Overview

This system uses BERT (Bidirectional Encoder Representations from Transformers) to:
1. Clean and preprocess question-tag datasets
2. Train a multi-label classification model
3. Predict the top 3 most relevant tags for new questions

## Features

- **Data Cleaning**: Robust text cleaning, HTML removal, tag parsing
- **BERT Integration**: Uses pre-trained BERT for contextual understanding
- **Multi-label Classification**: Predicts multiple tags per question
- **Top-K Prediction**: Returns top 3 tags with confidence scores
- **Flexible Architecture**: Configurable model parameters
- **Complete Pipeline**: End-to-end training and inference

## Project Structure

```
.
├── requirements.txt           # Python dependencies
├── data_cleaner.py           # Data cleaning and preprocessing
├── bert_model.py             # BERT model architecture
├── dataset.py                # PyTorch dataset implementation
├── trainer.py                # Training logic
├── inference.py              # Inference and prediction
├── train_pipeline.py         # Complete training pipeline
└── README_TAG_PREDICTION.md  # This file
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; import transformers; print('Installation successful!')"
```

## Quick Start

### Option 1: Use Sample Data (for testing)

```bash
# Train with automatically generated sample data
python train_pipeline.py --num_samples 1000 --num_epochs 3
```

### Option 2: Use Your Own Data

```bash
# Train with your CSV file
python train_pipeline.py \
    --data_path your_data.csv \
    --question_column question \
    --tag_column tags \
    --num_epochs 5 \
    --batch_size 16
```

## Data Format

Your CSV file should have at least two columns:

| question | tags |
|----------|------|
| How do I read a file in Python? | python,file-io,programming |
| What is REST API? | api,rest,web-services |

**Supported tag formats:**
- Comma-separated: `python,file-io,programming`
- Pipe-separated: `python|file-io|programming`
- List format: `['python', 'file-io', 'programming']`

## Usage

### 1. Training

#### Basic Training
```bash
python train_pipeline.py --data_path data.csv
```

#### Advanced Training with Custom Parameters
```bash
python train_pipeline.py \
    --data_path data.csv \
    --question_column question_text \
    --tag_column question_tags \
    --bert_model bert-base-uncased \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --max_length 256 \
    --min_tag_frequency 10 \
    --output_dir my_model
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | sample_data.csv | Path to input CSV file |
| `--question_column` | question | Name of question column |
| `--tag_column` | tags | Name of tag column |
| `--output_dir` | output | Output directory |
| `--bert_model` | bert-base-uncased | BERT model name |
| `--batch_size` | 16 | Training batch size |
| `--num_epochs` | 5 | Number of epochs |
| `--learning_rate` | 2e-5 | Learning rate |
| `--max_length` | 128 | Max sequence length |
| `--min_tag_frequency` | 5 | Min tag occurrences |
| `--hidden_dim` | 256 | Hidden layer size |
| `--dropout` | 0.3 | Dropout rate |

### 2. Inference

#### Interactive Mode
```bash
python inference.py output/checkpoints/best_model.pt output/tag_vocabulary.json
```

Then enter questions interactively:
```
Question: How do I sort an array in Python?

Top 3 Predicted Tags:
----------------------------------------
1. python               (confidence: 0.9234)
2. arrays               (confidence: 0.8567)
3. sorting              (confidence: 0.7891)
```

#### Programmatic Usage

```python
from inference import TagPredictor

# Load model
predictor = TagPredictor(
    model_path='output/checkpoints/best_model.pt',
    vocab_path='output/tag_vocabulary.json'
)

# Predict for single question
question = "How do I connect to a MySQL database?"
predictions = predictor.predict(question, top_k=3)

for tag, score in predictions:
    print(f"{tag}: {score:.4f}")

# Predict for multiple questions
questions = [
    "How to use async/await in JavaScript?",
    "What is object-oriented programming?"
]
batch_predictions = predictor.predict_batch(questions, top_k=3)
```

### 3. Individual Module Testing

#### Test Data Cleaner
```bash
python data_cleaner.py
```

#### Test BERT Model
```bash
python bert_model.py
```

#### Test Dataset
```bash
python dataset.py
```

## Model Architecture

```
Input Question
    ↓
BERT Tokenizer (max_length=128)
    ↓
BERT Encoder (bert-base-uncased)
    ↓
[CLS] Token Representation (768-dim)
    ↓
Linear Layer (768 → 256)
    ↓
ReLU + Dropout
    ↓
Linear Layer (256 → 128)
    ↓
ReLU + Dropout
    ↓
Linear Layer (128 → num_tags)
    ↓
Sigmoid Activation
    ↓
Top-3 Tags with Confidence Scores
```

## Training Process

1. **Data Cleaning**
   - Remove HTML tags and URLs
   - Clean special characters
   - Parse and normalize tags
   - Filter by minimum tag frequency

2. **Tokenization**
   - BERT tokenization with padding/truncation
   - Maximum sequence length: 128 tokens

3. **Model Training**
   - Loss: Binary Cross-Entropy with Logits
   - Optimizer: AdamW with weight decay
   - Learning rate scheduler: Linear warmup
   - Gradient clipping: max_norm=1.0

4. **Evaluation**
   - Metrics: Precision, Recall, F1-Score
   - Top-K Accuracy
   - Best model saved based on validation loss

## Output Files

After training, the following files are created:

```
output/
├── cleaned_data.csv                    # Cleaned dataset
├── tag_vocabulary.json                 # Tag mappings
└── checkpoints/
    ├── best_model.pt                   # Best model checkpoint
    ├── final_model.pt                  # Final model
    ├── checkpoint_epoch_1.pt           # Epoch checkpoints
    ├── checkpoint_epoch_2.pt
    └── training_history.json           # Training metrics
```

## Performance Tips

### For Better Accuracy
- Increase `--num_epochs` (10-20 epochs)
- Use larger BERT model: `--bert_model bert-large-uncased`
- Increase `--max_length` for longer questions
- Lower `--min_tag_frequency` to include more tags

### For Faster Training
- Reduce `--batch_size` if out of memory
- Use smaller BERT: `--bert_model distilbert-base-uncased`
- Reduce `--max_length`
- Use fewer epochs

### For Large Datasets
- Increase `--min_tag_frequency` to reduce tag vocabulary
- Use `--max_tags_per_question` to limit tags
- Increase `--batch_size` if you have GPU memory

## Example Workflows

### Workflow 1: Quick Test
```bash
# Generate sample data and train quickly
python train_pipeline.py --num_samples 500 --num_epochs 2 --batch_size 8
```

### Workflow 2: Production Training
```bash
# Train on real data with optimal settings
python train_pipeline.py \
    --data_path stackoverflow_data.csv \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --min_tag_frequency 50 \
    --output_dir production_model
```

### Workflow 3: Fine-tuning
```bash
# Train with custom BERT model
python train_pipeline.py \
    --data_path domain_specific_data.csv \
    --bert_model bert-large-uncased \
    --num_epochs 15 \
    --max_length 256
```

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python train_pipeline.py --batch_size 8

# Or use gradient accumulation (modify trainer.py)
```

### CUDA Not Available
```bash
# The code automatically falls back to CPU
# Training will be slower but will work
```

### Poor Predictions
- Ensure sufficient training data (>1000 samples)
- Increase training epochs
- Check data quality and tag consistency
- Increase `min_tag_frequency` to focus on common tags

## Advanced Usage

### Custom Data Cleaning

```python
from data_cleaner import DataCleaner
import pandas as pd

df = pd.read_csv('your_data.csv')

cleaner = DataCleaner(
    min_tag_frequency=10,
    max_tags_per_question=5
)

df_clean = cleaner.clean_dataset(df)
cleaner.save_vocabulary('custom_vocab.json')
```

### Custom Model Architecture

```python
from bert_model import BERTTagPredictor

model = BERTTagPredictor(
    num_tags=100,
    bert_model_name='bert-large-uncased',
    dropout=0.4,
    hidden_dim=512
)
```

### Batch Prediction

```python
from inference import TagPredictor

predictor = TagPredictor(
    model_path='output/checkpoints/best_model.pt',
    vocab_path='output/tag_vocabulary.json'
)

questions = [
    "How to use pandas in Python?",
    "What is machine learning?",
    # ... more questions
]

predictions = predictor.predict_batch(questions, top_k=3, batch_size=32)

for question, preds in zip(questions, predictions):
    print(f"\nQuestion: {question}")
    print("Tags:", [tag for tag, score in preds])
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU acceleration)

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this code, please cite:

```
BERT-based Tag Prediction System
https://github.com/yourusername/bert-tag-prediction
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example workflows
3. Examine the code comments in each module

## Future Enhancements

- [ ] Support for hierarchical tags
- [ ] Multi-language support
- [ ] Active learning for tag suggestion
- [ ] Web API for inference
- [ ] Model compression and optimization
- [ ] Transfer learning from domain-specific models
