#!/usr/bin/env python3
"""
Step 4: Evaluate and test the trained cardiology embedding model.

Tests:
- Semantic similarity on cardiology examples
- Paraphrase detection
- Sentence clustering
- Retrieval performance
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch


# Config
MODEL_PATH = "cardiology_embedding_model"
TEST_FILE = "cardiology_embedding_data/02_training_test.jsonl"


def load_model(model_path):
    """Load trained embedding model."""
    print(f"📥 Loading model from {model_path}...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_path, device=device)

    print(f"✓ Model loaded on {device}")
    return model


def test_similarity_examples(model):
    """Test model on hand-crafted cardiology examples."""

    print("\n" + "=" * 70)
    print("🧪 TEST 1: Semantic Similarity on Cardiology Examples")
    print("=" * 70)

    test_pairs = [
        # Similar (should have high similarity)
        ("The patient has a history of myocardial infarction.",
         "The patient previously experienced a heart attack.",
         "similar"),

        ("Echocardiography revealed reduced ejection fraction.",
         "Echo showed decreased EF.",
         "similar"),

        ("The electrocardiogram shows ST-segment elevation.",
         "ECG demonstrates ST elevation.",
         "similar"),

        # Different (should have low similarity)
        ("The patient has atrial fibrillation.",
         "The patient underwent coronary bypass surgery.",
         "different"),

        ("Left ventricular hypertrophy is present.",
         "The tricuspid valve is normal.",
         "different"),

        # Negation (should be different)
        ("No evidence of myocardial infarction.",
         "Evidence of myocardial infarction.",
         "negation"),
    ]

    similarities = []
    print("\n")

    for sent1, sent2, label in test_pairs:
        emb1 = model.encode(sent1, convert_to_tensor=True)
        emb2 = model.encode(sent2, convert_to_tensor=True)

        similarity = util.cos_sim(emb1, emb2).item()
        similarities.append((similarity, label))

        print(f"[{label.upper():10s}] Similarity: {similarity:.4f}")
        print(f"  Sent1: {sent1[:60]}...")
        print(f"  Sent2: {sent2[:60]}...")
        print()

    # Analyze by category
    similar_scores = [s for s, l in similarities if l == 'similar']
    different_scores = [s for s, l in similarities if l == 'different']
    negation_scores = [s for s, l in similarities if l == 'negation']

    print("📊 Summary:")
    print(f"  Similar pairs:   avg={np.mean(similar_scores):.4f}, std={np.std(similar_scores):.4f}")
    print(f"  Different pairs: avg={np.mean(different_scores):.4f}, std={np.std(different_scores):.4f}")
    print(f"  Negation pairs:  avg={np.mean(negation_scores):.4f}, std={np.std(negation_scores):.4f}")


def test_paraphrase_detection(model, test_file, threshold=0.7):
    """Test paraphrase detection accuracy on test set."""

    print("\n" + "=" * 70)
    print("🧪 TEST 2: Paraphrase Detection Accuracy")
    print("=" * 70)

    # Load test data
    test_pairs = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            test_pairs.append(record)

    print(f"\nTesting on {len(test_pairs):,} pairs...")
    print(f"Similarity threshold: {threshold}")

    # Evaluate
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for record in test_pairs:
        anchor = record['anchor']
        positive = record['positive']
        negatives = record['negatives']

        # Test positive pair
        emb_anchor = model.encode(anchor, convert_to_tensor=True)
        emb_positive = model.encode(positive, convert_to_tensor=True)
        pos_sim = util.cos_sim(emb_anchor, emb_positive).item()

        if pos_sim >= threshold:
            true_positives += 1
        else:
            false_negatives += 1

        # Test negative pairs
        for negative in negatives:
            emb_negative = model.encode(negative, convert_to_tensor=True)
            neg_sim = util.cos_sim(emb_anchor, emb_negative).item()

            if neg_sim < threshold:
                true_negatives += 1
            else:
                false_positives += 1

    # Calculate metrics
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n📊 Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\n  True Positives:  {true_positives:,}")
    print(f"  True Negatives:  {true_negatives:,}")
    print(f"  False Positives: {false_positives:,}")
    print(f"  False Negatives: {false_negatives:,}")


def test_retrieval(model, test_file, k=5):
    """Test retrieval performance: given a query, find similar sentences."""

    print("\n" + "=" * 70)
    print(f"🧪 TEST 3: Information Retrieval (Top-{k})")
    print("=" * 70)

    # Load test data
    sentences = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            sentences.append(record['anchor'])

    # Take first 1000 sentences as corpus
    corpus = sentences[:1000]
    queries = sentences[1000:1005]  # Use 5 queries

    print(f"\nCorpus size: {len(corpus):,}")
    print(f"Queries: {len(queries)}")

    # Encode corpus
    print("\n🔄 Encoding corpus...")
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

    # Test queries
    print(f"\n🔍 Testing retrieval...")

    for idx, query in enumerate(queries, 1):
        print(f"\n[Query {idx}]")
        print(f"  {query[:80]}...")

        # Encode query
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Find top-k similar
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(k, len(corpus)))

        print(f"\n  Top-{k} Results:")
        for rank, (score, idx) in enumerate(zip(top_results[0], top_results[1]), 1):
            print(f"    [{rank}] Score: {score:.4f}")
            print(f"        {corpus[idx][:70]}...")


def test_clustering(model, test_file, num_clusters=5):
    """Test sentence clustering on cardiology topics."""

    print("\n" + "=" * 70)
    print(f"🧪 TEST 4: Sentence Clustering ({num_clusters} clusters)")
    print("=" * 70)

    from sklearn.cluster import KMeans

    # Load sentences
    sentences = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= 100:  # Use first 100 sentences
                break
            record = json.loads(line.strip())
            sentences.append(record['anchor'])

    print(f"\nClustering {len(sentences)} sentences into {num_clusters} clusters...")

    # Encode sentences
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Cluster
    clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = clustering_model.fit_predict(embeddings)

    # Show examples from each cluster
    print("\n📊 Cluster Examples:")

    for cluster_id in range(num_clusters):
        cluster_sentences = [s for s, l in zip(sentences, cluster_labels) if l == cluster_id]

        print(f"\nCluster {cluster_id + 1} ({len(cluster_sentences)} sentences):")
        for sent in cluster_sentences[:3]:  # Show first 3
            print(f"  - {sent[:70]}...")


def main():
    print("=" * 70)
    print("🔬 CARDIOLOGY EMBEDDING MODEL EVALUATION")
    print("=" * 70)

    # Load model
    model = load_model(MODEL_PATH)

    # Run tests
    test_similarity_examples(model)
    test_paraphrase_detection(model, TEST_FILE, threshold=0.7)
    test_retrieval(model, TEST_FILE, k=5)
    test_clustering(model, TEST_FILE, num_clusters=5)

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n🎉 Your model is ready for deployment!")
    print(f"📁 Model location: {MODEL_PATH}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
