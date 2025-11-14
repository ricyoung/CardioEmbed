#!/usr/bin/env python3
"""
Step 2: Prepare training data for embedding model.

Creates:
- Training pairs (original, paraphrase)
- Hard negatives (similar but semantically different)
- Train/validation/test splits
- Different difficulty levels (easy/medium/hard)
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


# Config
INPUT_FILE = "cardiology_embedding_data/01_merged_paraphrases.jsonl"
OUTPUT_DIR = "cardiology_embedding_data"

TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.10
TEST_SPLIT = 0.05

RANDOM_SEED = 42


def load_merged_data(filepath):
    """Load merged paraphrase data."""
    print(f"\n📖 Loading merged data from {filepath}...")

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"✓ Loaded {len(data):,} paraphrase pairs")
    return data


def categorize_by_difficulty(data):
    """Categorize paraphrases by difficulty/similarity level."""

    print("\n📊 Categorizing by difficulty...")

    categories = {
        'easy': [],      # Conservative paraphrases (Ministral)
        'medium': [],    # Moderate changes
        'hard': []       # Creative paraphrases (GLM)
    }

    for record in data:
        original = record['original']
        paraphrase = record['paraphrase']
        model = record.get('model', 'unknown')

        # Simple heuristic: length ratio and model type
        len_ratio = len(paraphrase) / max(len(original), 1)

        # Conservative paraphrases are usually similar length
        if 'ministral' in model.lower() or 0.85 <= len_ratio <= 1.15:
            categories['easy'].append(record)
        # Creative paraphrases often change length more
        elif 'glm' in model.lower() or len_ratio < 0.7 or len_ratio > 1.3:
            categories['hard'].append(record)
        else:
            categories['medium'].append(record)

    print(f"  Easy:   {len(categories['easy']):,}")
    print(f"  Medium: {len(categories['medium']):,}")
    print(f"  Hard:   {len(categories['hard']):,}")

    return categories


def create_hard_negatives(data, num_negatives=3):
    """Create hard negative examples for each paraphrase pair.

    Hard negatives are sentences that are:
    - From the same domain (cardiology)
    - Structurally similar
    - But semantically different
    """

    print(f"\n🔍 Creating hard negatives (top-{num_negatives} per pair)...")

    # Extract all original sentences
    all_originals = [record['original'] for record in data]

    pairs_with_negatives = []

    for idx, record in enumerate(data):
        original = record['original']
        paraphrase = record['paraphrase']

        # Find hard negatives: other sentences with similar length
        original_len = len(original.split())

        # Get sentences with similar length (±30%)
        similar_length = []
        for other_idx, other_orig in enumerate(all_originals):
            if other_idx == idx:
                continue

            other_len = len(other_orig.split())
            if 0.7 * original_len <= other_len <= 1.3 * original_len:
                similar_length.append(other_orig)

        # Randomly sample negatives
        if len(similar_length) >= num_negatives:
            negatives = random.sample(similar_length, num_negatives)
        else:
            # If not enough similar, sample from all
            negatives = random.sample(
                [s for i, s in enumerate(all_originals) if i != idx],
                min(num_negatives, len(all_originals) - 1)
            )

        pairs_with_negatives.append({
            'anchor': original,
            'positive': paraphrase,
            'negatives': negatives,
            'metadata': record['metadata'],
            'model': record['model'],
            'source': record['source']
        })

    print(f"✓ Created {len(pairs_with_negatives):,} pairs with hard negatives")
    return pairs_with_negatives


def split_dataset(data, train_ratio, val_ratio, test_ratio, seed=42):
    """Split data into train/validation/test sets."""

    print(f"\n📑 Splitting dataset: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")

    random.seed(seed)
    np.random.seed(seed)

    # First split: train vs (val+test)
    train_data, temp_data = train_test_split(
        data,
        test_size=(val_ratio + test_ratio),
        random_state=seed
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_ratio_adjusted),
        random_state=seed
    )

    print(f"  Train: {len(train_data):,}")
    print(f"  Val:   {len(val_data):,}")
    print(f"  Test:  {len(test_data):,}")

    return train_data, val_data, test_data


def save_split(data, split_name, output_dir):
    """Save dataset split to file."""

    output_file = f"{output_dir}/02_training_{split_name}.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')

    print(f"  Saved {split_name}: {output_file}")
    return output_file


def create_sentence_transformers_format(data, split_name, output_dir):
    """Create simplified format for sentence-transformers library.

    Format: CSV with columns [sentence1, sentence2, label]
    - label=1: positive pairs (paraphrases)
    - label=0: negative pairs
    """

    output_file = f"{output_dir}/02_training_{split_name}_simple.csv"

    import csv

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence1', 'sentence2', 'label'])

        for record in data:
            anchor = record['anchor']
            positive = record['positive']
            negatives = record['negatives']

            # Positive pair
            writer.writerow([anchor, positive, 1])

            # Negative pairs (use first negative only for simple format)
            if negatives:
                writer.writerow([anchor, negatives[0], 0])

    print(f"  Saved simple format: {output_file}")
    return output_file


def generate_statistics(train_data, val_data, test_data, output_dir):
    """Generate statistics about the prepared dataset."""

    stats_file = f"{output_dir}/02_training_statistics.txt"

    def compute_stats(data, split_name):
        stats = {
            'total': len(data),
            'models': defaultdict(int),
            'sources': defaultdict(int),
            'avg_anchor_len': 0,
            'avg_positive_len': 0,
            'avg_negative_len': 0
        }

        anchor_lens = []
        positive_lens = []
        negative_lens = []

        for record in data:
            stats['models'][record['model']] += 1
            stats['sources'][record['source']] += 1

            anchor_lens.append(len(record['anchor'].split()))
            positive_lens.append(len(record['positive'].split()))
            if record['negatives']:
                negative_lens.extend([len(neg.split()) for neg in record['negatives']])

        stats['avg_anchor_len'] = np.mean(anchor_lens)
        stats['avg_positive_len'] = np.mean(positive_lens)
        stats['avg_negative_len'] = np.mean(negative_lens) if negative_lens else 0

        return stats

    train_stats = compute_stats(train_data, 'train')
    val_stats = compute_stats(val_data, 'val')
    test_stats = compute_stats(test_data, 'test')

    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("TRAINING DATA STATISTICS\n")
        f.write("=" * 70 + "\n\n")

        for split_name, stats in [('TRAIN', train_stats), ('VALIDATION', val_stats), ('TEST', test_stats)]:
            f.write(f"{split_name} Split:\n")
            f.write(f"  Total pairs: {stats['total']:,}\n")
            f.write(f"  Avg anchor length: {stats['avg_anchor_len']:.1f} words\n")
            f.write(f"  Avg positive length: {stats['avg_positive_len']:.1f} words\n")
            f.write(f"  Avg negative length: {stats['avg_negative_len']:.1f} words\n")

            f.write(f"  Models:\n")
            for model, count in sorted(stats['models'].items()):
                f.write(f"    {model}: {count:,}\n")

            f.write(f"  Sources:\n")
            for source, count in sorted(stats['sources'].items()):
                f.write(f"    {source}: {count:,}\n")

            f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"\n📊 Statistics saved: {stats_file}")


def main():
    print("=" * 70)
    print("🎯 PREPARING TRAINING DATA FOR EMBEDDING MODEL")
    print("=" * 70)

    # Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    data = load_merged_data(INPUT_FILE)

    # Categorize by difficulty
    categories = categorize_by_difficulty(data)

    # Create hard negatives
    data_with_negatives = create_hard_negatives(data, num_negatives=3)

    # Split dataset
    train_data, val_data, test_data = split_dataset(
        data_with_negatives,
        TRAIN_SPLIT,
        VAL_SPLIT,
        TEST_SPLIT,
        seed=RANDOM_SEED
    )

    # Save splits
    print("\n💾 Saving dataset splits...")
    train_file = save_split(train_data, 'train', OUTPUT_DIR)
    val_file = save_split(val_data, 'val', OUTPUT_DIR)
    test_file = save_split(test_data, 'test', OUTPUT_DIR)

    # Create simple format for sentence-transformers
    print("\n📝 Creating sentence-transformers format...")
    create_sentence_transformers_format(train_data, 'train', OUTPUT_DIR)
    create_sentence_transformers_format(val_data, 'val', OUTPUT_DIR)
    create_sentence_transformers_format(test_data, 'test', OUTPUT_DIR)

    # Generate statistics
    generate_statistics(train_data, val_data, test_data, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("✅ TRAINING DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"  Train: {len(train_data):,} pairs")
    print(f"  Val:   {len(val_data):,} pairs")
    print(f"  Test:  {len(test_data):,} pairs")
    print("\n🚀 Ready for model training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
