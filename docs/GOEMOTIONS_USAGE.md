# GoEmotions Dataset Usage Guide

This guide explains how to use the GoEmotions dataset for training text-based emotion classifiers in the Relaxation Agent project.

## Overview

GoEmotions is a large-scale dataset of Reddit comments labeled with 27 emotion categories. We've integrated it into the project to improve text-based emotion classification.

## Installation

First, install the required dependency:

```bash
pip install datasets
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Downloading GoEmotions

### Option 1: Download via Script (Recommended)

```bash
python scripts/download_goemotions.py --output_dir ./data/goemotions
```

This will download all splits (train, validation, test) to the specified directory.

### Option 2: Download Specific Splits

```bash
python scripts/download_goemotions.py --output_dir ./data/goemotions --splits train validation
```

### Option 3: Download with Limits (for testing)

```bash
python scripts/download_goemotions.py --output_dir ./data/goemotions --max_samples 1000
```

### Options

- `--output_dir`: Directory to save downloaded data (default: `./data/goemotions`)
- `--splits`: Which splits to download (default: `train validation test`)
- `--simplified`: Use simplified version with 54k rows (default: True)
- `--no-simplified`: Use full version with 211k rows
- `--max_samples`: Maximum samples per split (useful for testing)

## Emotion Mapping

GoEmotions has 27 emotion categories, which we map to our 10 project emotion labels:

| GoEmotions Categories | Project Emotion |
|----------------------|-----------------|
| admiration, amusement, approval, caring, curiosity, desire, excitement, gratitude, joy, love, optimism, pride, relief | **happy** |
| anger, annoyance, disapproval | **angry** |
| disappointment, embarrassment, grief, remorse, sadness | **sad** |
| disgust | **disgusted** |
| fear | **fearful** |
| nervousness | **anxious** |
| surprise, realization | **surprised** |
| neutral | **neutral** |

## Training Text Emotion Classifier

### Basic Training

```bash
python -m src.milestone_c.train_text --data_dir ./data/goemotions --from_huggingface
```

This will:
1. Load GoEmotions from Hugging Face
2. Preprocess the text
3. Train a DistilBERT-based emotion classifier
4. Save the model to `./models/text_emotion_classifier.pt`

### Training from Local Files

If you've already downloaded the dataset:

```bash
python -m src.milestone_c.train_text --data_dir ./data/goemotions --split train
```

### Training Options

- `--data_dir`: Path to GoEmotions data directory
- `--from_huggingface`: Load directly from Hugging Face (requires internet)
- `--split`: Dataset split to use (`train`, `validation`, `test`)
- `--output_dir`: Output directory for trained model (default: `./models`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 2e-5)
- `--device`: Device to use (`cpu` or `cuda`)
- `--max_samples`: Maximum samples to use (for testing)
- `--simplified`: Use simplified GoEmotions version (default: True)

### Example: Quick Test Run

```bash
python -m src.milestone_c.train_text \
    --data_dir ./data/goemotions \
    --from_huggingface \
    --max_samples 1000 \
    --epochs 3 \
    --batch_size 16
```

## Using GoEmotions in Code

### Load from Hugging Face

```python
from src.milestone_b.goemotions_loader import GoEmotionsLoader

loader = GoEmotionsLoader()
df = loader.load_from_huggingface(
    split="train",
    simplified=True,
    max_samples=1000
)
```

### Load from Local Files

```python
from src.milestone_b.goemotions_loader import GoEmotionsLoader

loader = GoEmotionsLoader()
df = loader.load_from_local("./data/goemotions/goemotions_train.csv")
```

### Using with DatasetLoader

```python
from src.milestone_b.audio_features import DatasetLoader

loader = DatasetLoader("./data/goemotions")

# Load from Hugging Face
df = loader.load_goemotions_dataset(
    from_huggingface=True,
    split="train",
    simplified=True
)

# Or load from local
df = loader.load_goemotions_dataset(
    dataset_path="./data/goemotions/goemotions_train.csv",
    from_huggingface=False
)

# Preprocess for training
preprocessed_texts, labels = loader.preprocess_text_dataset(df)
```

## Dataset Statistics

### Simplified Version
- **Train**: 43,410 samples
- **Validation**: 5,426 samples
- **Test**: 5,426 samples
- **Total**: ~54,000 samples

### Full Version
- **Train**: ~173,000 samples
- **Validation**: ~21,000 samples
- **Test**: ~21,000 samples
- **Total**: ~211,000 samples

## Integration with Existing Pipeline

The GoEmotions-trained model can be used with the existing text emotion classifier:

```python
from src.core.emotion_classifier import EmotionClassifier

# Load trained model
classifier = EmotionClassifier(model_path="./models/text_emotion_classifier.pt")

# Use for prediction
emotion_probs = classifier.predict_from_text("I'm feeling really stressed!")
```

## Troubleshooting

### Import Error: datasets library not found

```bash
pip install datasets
```

### Out of Memory

- Use `--max_samples` to limit dataset size
- Use `--simplified` flag (default)
- Reduce `--batch_size`
- Use CPU instead of GPU: `--device cpu`

### Slow Download

- The dataset is large (~100MB+)
- Ensure stable internet connection
- Consider downloading to a local directory first

### Emotion Mapping Issues

If you need to customize emotion mappings, edit:
- `src/milestone_b/goemotions_loader.py`
- `GOEMOTIONS_TO_PROJECT_EMOTIONS` dictionary

## Next Steps

1. **Download the dataset**: `python scripts/download_goemotions.py`
2. **Train the model**: `python -m src.milestone_c.train_text --from_huggingface`
3. **Evaluate**: Test the trained model on validation/test sets
4. **Integrate**: Use the trained model in the Relaxation Agent pipeline

## References

- [GoEmotions Paper](https://arxiv.org/abs/2005.00547)
- [Hugging Face Dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- [GitHub Repository](https://github.com/google-research/google-research/tree/master/goemotions)
