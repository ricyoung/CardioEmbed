#!/usr/bin/env python3
"""
Step 3: Train cardiology embedding model using sentence-transformers.

Models tested:
- PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
- BioBERT (dmis-lab/biobert-v1.1)
- SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)

Training objective: Multiple Negatives Ranking Loss (best for paraphrase pairs)
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models
)
from torch.utils.data import DataLoader


# Configuration
BASE_MODEL = "Qwen/Qwen3-8B"
# Alternative models:
# BASE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# BASE_MODEL = "dmis-lab/biobert-v1.1"
# BASE_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

TRAIN_FILE = "cardiology_embedding_data/02_training_train.jsonl"
VAL_FILE = "cardiology_embedding_data/02_training_val.jsonl"
TEST_FILE = "cardiology_embedding_data/02_training_test.jsonl"

OUTPUT_DIR = "cardiology_embedding_model_qwen3"
LOG_DIR = f"{OUTPUT_DIR}/logs"

# Training hyperparameters
BATCH_SIZE = 2  # Reduced for Qwen3-8B (8B params) on 24GB GPU
EPOCHS = 3
WARMUP_STEPS = 1000
MAX_SEQ_LENGTH = 128  # Reduced from 256 to save memory

# Learning rate
LEARNING_RATE = 2e-5

# Evaluation
EVAL_STEPS = 500  # Evaluate every N steps


def load_training_data(filepath, max_samples=None):
    """Load training data from JSONL file."""

    print(f"📖 Loading {filepath}...")

    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break

            record = json.loads(line.strip())

            # Create positive pair
            examples.append(
                InputExample(texts=[record['anchor'], record['positive']])
            )

    print(f"  Loaded {len(examples):,} training examples")
    return examples


def create_evaluator(val_file):
    """Create evaluator for validation set."""

    print(f"📊 Creating evaluator from {val_file}...")

    # Load validation data
    sentences1 = []
    sentences2 = []
    scores = []

    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())

            anchor = record['anchor']
            positive = record['positive']

            # Positive pair (similarity = 1.0)
            sentences1.append(anchor)
            sentences2.append(positive)
            scores.append(1.0)

            # Negative pair (similarity = 0.0)
            if record['negatives']:
                sentences1.append(anchor)
                sentences2.append(record['negatives'][0])
                scores.append(0.0)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1,
        sentences2,
        scores,
        name='cardiology_val',
        show_progress_bar=True
    )

    print(f"  Created evaluator with {len(sentences1):,} pairs")
    return evaluator


def train_model(base_model_name, train_data, evaluator, output_dir):
    """Train the embedding model."""

    print("=" * 70)
    print("🚀 TRAINING CARDIOLOGY EMBEDDING MODEL")
    print("=" * 70)
    print(f"\nBase model: {base_model_name}")
    print(f"Training examples: {len(train_data):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")

    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load base model
    print(f"\n📥 Loading base model...")
    model = SentenceTransformer(base_model_name, device=device)

    # Set max sequence length
    model.max_seq_length = MAX_SEQ_LENGTH

    # Create dataloader
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    # Define loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"\n🎯 Training objectives:")
    print(f"  Loss: Multiple Negatives Ranking Loss")
    print(f"  Strategy: Contrastive learning with in-batch negatives")

    # Training arguments
    warmup_steps = min(WARMUP_STEPS, len(train_dataloader) * EPOCHS // 10)

    print(f"\n⚙️  Training configuration:")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Evaluation steps: {EVAL_STEPS}")
    print(f"  Output directory: {output_dir}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Train
    print(f"\n🏃 Starting training...\n")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        evaluation_steps=EVAL_STEPS,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': LEARNING_RATE},
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True
    )

    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {output_dir}")

    return model


def evaluate_on_test(model, test_file):
    """Evaluate trained model on test set."""

    print("\n" + "=" * 70)
    print("🧪 EVALUATING ON TEST SET")
    print("=" * 70)

    # Load test data
    sentences1 = []
    sentences2 = []
    scores = []

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())

            anchor = record['anchor']
            positive = record['positive']

            # Positive pair
            sentences1.append(anchor)
            sentences2.append(positive)
            scores.append(1.0)

            # Negative pair
            if record['negatives']:
                sentences1.append(anchor)
                sentences2.append(record['negatives'][0])
                scores.append(0.0)

    test_evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1,
        sentences2,
        scores,
        name='cardiology_test',
        show_progress_bar=True
    )

    # Evaluate
    test_score = test_evaluator(model, output_path=OUTPUT_DIR)

    print(f"\n📊 Test Set Performance:")
    print(f"  Spearman Correlation: {test_score:.4f}")

    return test_score


def save_training_info(base_model, output_dir):
    """Save training configuration and metadata."""

    info = {
        'base_model': base_model,
        'training_date': datetime.now().isoformat(),
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'warmup_steps': WARMUP_STEPS,
            'max_seq_length': MAX_SEQ_LENGTH
        },
        'loss_function': 'MultipleNegativesRankingLoss',
        'evaluation_metric': 'EmbeddingSimilarity'
    }

    info_file = f"{output_dir}/training_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    print(f"\n📝 Training info saved: {info_file}")


def main():
    print("=" * 70)
    print("💉 CARDIOLOGY EMBEDDING MODEL TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load training data
    train_data = load_training_data(TRAIN_FILE)

    # Create validation evaluator
    val_evaluator = create_evaluator(VAL_FILE)

    # Train model
    model = train_model(
        BASE_MODEL,
        train_data,
        val_evaluator,
        OUTPUT_DIR
    )

    # Evaluate on test set
    test_score = evaluate_on_test(model, TEST_FILE)

    # Save training info
    save_training_info(BASE_MODEL, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("✅ ALL COMPLETE!")
    print("=" * 70)
    print(f"\n🎉 Your cardiology embedding model is ready!")
    print(f"📁 Model location: {OUTPUT_DIR}/")
    print(f"📊 Test score: {test_score:.4f}")
    print("\n💡 Next steps:")
    print("  1. Test the model with example queries")
    print("  2. Deploy to your application")
    print("  3. Monitor performance on real data")
    print("=" * 70)


if __name__ == "__main__":
    main()
