#!/usr/bin/env python3
"""
Step 1: Merge all paraphrase sources and prepare unified dataset.

Merges:
- RTX 6000 results (Qwen3-32B): 3,471 paraphrases
- 4090 results (Qwen3-8B): 1,536 paraphrases
- API results (Ministral-8B + GLM-4-32B): ~154,699 paraphrases

Output: Unified JSONL with source tracking
"""

import json
import glob
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# Input paths
RTX6000_FILE = "remote_backup/repo/results/paraphrases_qwen3_32b_full.jsonl"
GPU_4090_FILE = "paraphrases_4090_qwen3_8b.jsonl"
API_MINISTRAL_FILE = "paraphrases_dual_model/paraphrases_ministral_8b.jsonl"
API_GLM_FILE = "paraphrases_dual_model/paraphrases_glm_4_32b.jsonl"

# Output
OUTPUT_DIR = "cardiology_embedding_data"
MERGED_FILE = f"{OUTPUT_DIR}/01_merged_paraphrases.jsonl"
STATS_FILE = f"{OUTPUT_DIR}/01_merge_statistics.txt"


def load_jsonl(filepath):
    """Load JSONL file and return list of records."""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping invalid JSON at {filepath}:{line_num}: {str(e)[:50]}")
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        return []

    return records


def normalize_record(record, source):
    """Normalize record format across different sources."""

    # Extract original and paraphrase
    original = record.get('original', record.get('sentence', ''))
    paraphrase = record.get('paraphrase', record.get('paraphrased', ''))

    # Clean text
    original = original.strip()
    paraphrase = paraphrase.strip()

    # Remove quotes if present
    paraphrase = paraphrase.strip('"').strip("'")

    # Extract model info
    model = record.get('model', 'unknown')

    return {
        'original': original,
        'paraphrase': paraphrase,
        'source': source,
        'model': model,
        'original_length': len(original),
        'paraphrase_length': len(paraphrase),
        'metadata': {
            'input_tokens': record.get('input_tokens', 0),
            'output_tokens': record.get('output_tokens', 0),
            'sentence_id': record.get('sentence_id', -1)
        }
    }


def is_valid_pair(record):
    """Check if paraphrase pair is valid for training."""

    original = record['original']
    paraphrase = record['paraphrase']

    # Both must be non-empty
    if not original or not paraphrase:
        return False, "empty_text"

    # Minimum length (avoid single words or very short fragments)
    if len(original.split()) < 3 or len(paraphrase.split()) < 3:
        return False, "too_short"

    # Maximum length (extremely long texts may be errors)
    if len(original.split()) > 500 or len(paraphrase.split()) > 500:
        return False, "too_long"

    # Paraphrase shouldn't be identical to original
    if original.lower() == paraphrase.lower():
        return False, "identical"

    # Avoid common non-medical headers/footers
    non_medical_patterns = [
        'TABLE OF CONTENTS',
        'CHAPTER ',
        'SECTION ',
        'PAGE ',
        'FIGURE ',
        'Copyright',
        '©',
        'All rights reserved',
        'INDEX',
        'APPENDIX',
        'REFERENCES',
        'PERSONAL PERSPECTIVES'
    ]

    for pattern in non_medical_patterns:
        if pattern in original.upper() or pattern in paraphrase.upper():
            # Allow SECTION if it has medical context
            if 'SECTION' in pattern and any(med in original.upper() for med in ['DOPPLER', 'ECHO', 'CARDIAC', 'VENTRICULAR']):
                continue
            return False, "non_medical_content"

    # Check if paraphrase is actually an error message
    error_indicators = [
        'does not appear to be',
        'cannot be provided',
        'not a medical sentence',
        'rather a title or heading'
    ]

    for indicator in error_indicators:
        if indicator.lower() in paraphrase.lower():
            return False, "error_message"

    return True, "valid"


def main():
    print("=" * 70)
    print("🔗 MERGING PARAPHRASE SOURCES")
    print("=" * 70)

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load all sources
    print("\n📖 Loading paraphrase sources...")

    sources = {
        'rtx6000_qwen32b': load_jsonl(RTX6000_FILE),
        '4090_qwen8b': load_jsonl(GPU_4090_FILE),
        'api_ministral8b': load_jsonl(API_MINISTRAL_FILE),
        'api_glm32b': load_jsonl(API_GLM_FILE),
    }

    # Statistics
    stats = {
        'total_loaded': 0,
        'by_source': defaultdict(int),
        'valid': 0,
        'filtered': defaultdict(int),
        'by_model': defaultdict(int)
    }

    # Process and merge
    print("\n🔄 Processing and merging...")

    all_records = []
    seen_originals = set()  # Track duplicates by original sentence

    for source_name, records in sources.items():
        print(f"\n  Processing {source_name}: {len(records):,} records")
        stats['total_loaded'] += len(records)
        stats['by_source'][source_name] = len(records)

        for record in records:
            # Normalize
            normalized = normalize_record(record, source_name)

            # Validate
            is_valid, reason = is_valid_pair(normalized)

            if is_valid:
                # Check for duplicate originals
                original_key = normalized['original'].lower().strip()

                if original_key in seen_originals:
                    stats['filtered']['duplicate_original'] += 1
                    continue

                seen_originals.add(original_key)
                all_records.append(normalized)
                stats['valid'] += 1
                stats['by_model'][normalized['model']] += 1
            else:
                stats['filtered'][reason] += 1

    # Sort by original sentence (for consistency)
    all_records.sort(key=lambda x: x['original'])

    # Write merged file
    print(f"\n💾 Writing merged dataset: {MERGED_FILE}")

    with open(MERGED_FILE, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record) + '\n')

    # Write statistics
    print(f"\n📊 Writing statistics: {STATS_FILE}")

    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("PARAPHRASE MERGE STATISTICS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Total Records Loaded: {stats['total_loaded']:,}\n")
        f.write(f"Valid Records: {stats['valid']:,}\n")
        f.write(f"Filtered Records: {sum(stats['filtered'].values()):,}\n\n")

        f.write("Records by Source:\n")
        for source, count in sorted(stats['by_source'].items()):
            f.write(f"  {source:20s}: {count:,}\n")

        f.write("\nRecords by Model:\n")
        for model, count in sorted(stats['by_model'].items()):
            f.write(f"  {model:20s}: {count:,}\n")

        f.write("\nFiltered Reasons:\n")
        for reason, count in sorted(stats['filtered'].items(), key=lambda x: -x[1]):
            f.write(f"  {reason:25s}: {count:,}\n")

        f.write("\n" + "=" * 70 + "\n")

    # Print summary
    print("\n" + "=" * 70)
    print("✅ MERGE COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Total loaded: {stats['total_loaded']:,}")
    print(f"  Valid pairs: {stats['valid']:,}")
    print(f"  Filtered: {sum(stats['filtered'].values()):,}")
    print(f"\n📁 Output: {MERGED_FILE}")
    print(f"📈 Statistics: {STATS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
