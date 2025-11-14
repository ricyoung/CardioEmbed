#!/usr/bin/env python3
"""
Usage example: How to use the trained cardiology embedding model.

Examples:
- Encode sentences to embeddings
- Compute semantic similarity
- Find most similar sentences
- Semantic search in cardiology text
"""

from sentence_transformers import SentenceTransformer, util
import torch


# Load trained model
MODEL_PATH = "cardiology_embedding_model"

print("=" * 70)
print("💉 CARDIOLOGY EMBEDDING MODEL - USAGE EXAMPLES")
print("=" * 70)

print(f"\n📥 Loading model from {MODEL_PATH}...")
model = SentenceTransformer(MODEL_PATH)
print("✓ Model loaded!")


# Example 1: Encode sentences
print("\n" + "=" * 70)
print("Example 1: Encoding Sentences")
print("=" * 70)

sentences = [
    "The patient presents with chest pain and shortness of breath.",
    "Echocardiography revealed reduced left ventricular ejection fraction.",
    "The electrocardiogram shows ST-segment elevation in leads II, III, and aVF."
]

print("\nSentences:")
for idx, sent in enumerate(sentences, 1):
    print(f"  {idx}. {sent}")

print("\n🔄 Encoding...")
embeddings = model.encode(sentences)

print(f"✓ Generated embeddings: shape={embeddings.shape}")
print(f"  {len(sentences)} sentences × {embeddings.shape[1]} dimensions")


# Example 2: Compute similarity between two sentences
print("\n" + "=" * 70)
print("Example 2: Computing Semantic Similarity")
print("=" * 70)

sent1 = "The patient has a history of myocardial infarction."
sent2 = "The patient previously experienced a heart attack."

print(f"\nSentence 1: {sent1}")
print(f"Sentence 2: {sent2}")

emb1 = model.encode(sent1, convert_to_tensor=True)
emb2 = model.encode(sent2, convert_to_tensor=True)

similarity = util.cos_sim(emb1, emb2).item()

print(f"\n📊 Cosine Similarity: {similarity:.4f}")

if similarity > 0.8:
    print("✅ Highly similar (paraphrases)")
elif similarity > 0.5:
    print("🟡 Moderately similar")
else:
    print("❌ Not similar")


# Example 3: Find most similar sentence in a corpus
print("\n" + "=" * 70)
print("Example 3: Semantic Search in Corpus")
print("=" * 70)

query = "What are the symptoms of heart failure?"

corpus = [
    "Heart failure symptoms include dyspnea, fatigue, and edema.",
    "Coronary angiography shows significant stenosis in the LAD.",
    "The patient underwent successful percutaneous coronary intervention.",
    "Common signs of congestive heart failure are shortness of breath and swelling.",
    "Echocardiogram demonstrated normal left ventricular function.",
]

print(f"\n🔍 Query: {query}")
print(f"\n📚 Searching corpus of {len(corpus)} sentences...")

# Encode query and corpus
query_embedding = model.encode(query, convert_to_tensor=True)
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Compute similarities
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

# Get top 3 results
top_k = 3
top_results = torch.topk(cos_scores, k=min(top_k, len(corpus)))

print(f"\n🎯 Top {top_k} Results:")
for rank, (score, idx) in enumerate(zip(top_results[0], top_results[1]), 1):
    print(f"\n  Rank {rank}: Score = {score:.4f}")
    print(f"  {corpus[idx]}")


# Example 4: Batch encoding for efficiency
print("\n" + "=" * 70)
print("Example 4: Batch Encoding (for large datasets)")
print("=" * 70)

large_corpus = [
    f"Example cardiology sentence {i}" for i in range(1000)
]

print(f"\n📦 Encoding {len(large_corpus):,} sentences in batches...")

embeddings = model.encode(
    large_corpus,
    batch_size=32,  # Process 32 sentences at a time
    show_progress_bar=True,
    normalize_embeddings=True  # For cosine similarity
)

print(f"✓ Encoded {len(embeddings):,} sentences")
print(f"  Shape: {embeddings.shape}")
print(f"  Memory: ~{embeddings.nbytes / 1e6:.1f} MB")


# Example 5: Save and load embeddings
print("\n" + "=" * 70)
print("Example 5: Saving Embeddings for Reuse")
print("=" * 70)

import numpy as np

print("\n💾 Saving embeddings to file...")
np.save('example_embeddings.npy', embeddings)
print("✓ Saved to: example_embeddings.npy")

print("\n📥 Loading embeddings from file...")
loaded_embeddings = np.load('example_embeddings.npy')
print(f"✓ Loaded: shape={loaded_embeddings.shape}")


# Example 6: Production API usage pattern
print("\n" + "=" * 70)
print("Example 6: Production Usage Pattern")
print("=" * 70)

print("""
# Typical production workflow:

class CardiologyEmbeddingService:
    def __init__(self, model_path='cardiology_embedding_model'):
        self.model = SentenceTransformer(model_path)

    def encode(self, texts, normalize=True):
        '''Encode texts to embeddings.'''
        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

    def similarity(self, text1, text2):
        '''Compute similarity between two texts.'''
        emb1, emb2 = self.encode([text1, text2])
        return np.dot(emb1, emb2)

    def find_similar(self, query, corpus, top_k=5):
        '''Find most similar texts in corpus.'''
        query_emb = self.encode([query])[0]
        corpus_embs = self.encode(corpus)

        # Compute similarities
        scores = np.dot(corpus_embs, query_emb)

        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [(corpus[i], scores[i]) for i in top_indices]

# Usage:
service = CardiologyEmbeddingService()

# Encode patient note
note = "Patient presents with chest pain..."
embedding = service.encode([note])[0]

# Find similar cases
similar_cases = service.find_similar(note, historical_cases, top_k=5)
""")


print("\n" + "=" * 70)
print("✅ EXAMPLES COMPLETE")
print("=" * 70)
print("""
🎉 Your cardiology embedding model is ready for production!

Key capabilities:
✓ Semantic understanding of cardiology text
✓ High-quality sentence embeddings
✓ Efficient batch processing
✓ Semantic search and similarity
✓ Paraphrase detection

Next steps:
1. Integrate into your application
2. Monitor performance on real data
3. Fine-tune on domain-specific tasks
4. Deploy with FastAPI or similar framework
""")
print("=" * 70)
