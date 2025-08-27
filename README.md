# Entity Recognition Training Project

This project trains a Qwen model for entity recognition from Business Intelligence questions.

## Project Structure

```
bi-model-intent/
├── data/
│   ├── training/                    # Training data
│   │   └── training_data_entity_recognition.json
│   ├── testing/                     # Test data
│   │   └── test_examples_101-200.json
│   └── prompts/                     # Prompt templates
│       └── entity_recognition_prompt.txt
├── scripts/                         # Training and utility scripts
│   ├── train_entity_recognition.py  # Main training script
│   ├── training_config.py           # Configuration
│   └── convert_to_simple_format.py  # Data format converter
├── models/                          # Trained models (created during training)
├── results/                         # Evaluation results (to be created)
├── requirements.txt                 # Python dependencies
└── entity_training.ipynb            # Jupyter notebook for training
```

## Quick Start

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Model
```bash
cd scripts
python train_entity_recognition.py
```

### 3. Change Model
Edit `MODEL_NAME` in `scripts/train_entity_recognition.py`:
```python
MODEL_NAME = "qwen-1.5b"  # Options: "qwen-1.5b", "qwen-0.5b"
```

## Data Format

### Training/Test Data Format
```json
{
  "input": "How many heads of the publishers are older than 56?",
  "output": {
    "dimensions": ["publisher", "publishers", "age"],
    "measures": ["heads"],
    "calculations": ["count"],
    "filters": ["older than 56"],
    "time_references": []
  }
}
```

## Available Models

- `qwen-1.5b`: Qwen/Qwen2.5-1.5B-Instruct (batch_size=4)
- `qwen-0.5b`: Qwen/Qwen2.5-0.5B (batch_size=8)

## Training Configuration

- **Learning Rate**: 2e-4
- **Epochs**: 3
- **LoRA**: r=16, alpha=32, dropout=0.1
- **Quantization**: 4-bit for memory efficiency

## Output

Trained models are saved to `./models/entity_recognition_model/` and automatically pushed to HuggingFace Hub under `ssuki/{model-name}-entity-recognition`.

## Next Steps

1. Run inference on test set
2. Measure performance metrics
3. Compare with baseline models
4. Benchmark inference time
