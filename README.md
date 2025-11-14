# CardioEmbed: Domain-Specialized Text Embeddings for Clinical Cardiology

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/richardyoung/CardioEmbed)

**Domain-specialized embedding model trained on comprehensive cardiology textbooks for clinical applications.**

> **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (To be published)
>
> **Authors:** Richard J. Young¹, Alice M. Matthews²
>
> **Affiliations:**
> ¹ University of Nevada Las Vegas, Department of Neuroscience
> ² Concorde Career College, Department of Cardiovascular and Medical Diagnostic Sonography

---

## 📄 Overview

CardioEmbed is a domain-specialized embedding model trained on comprehensive cardiology textbooks to capture the procedural knowledge and specialized terminology essential for clinical cardiology applications. Unlike existing biomedical embedding models trained primarily on research literature (PubMed), CardioEmbed is trained on clinical textbook content that better represents the knowledge used in actual clinical practice.

**Key Result:** CardioEmbed achieves **99.60% retrieval accuracy** on cardiac-specific semantic tasks, a **+15.94 percentage point improvement** over MedTE (the current state-of-the-art medical embedding model).

### Key Features

- **Domain-Specialized Training:** Fine-tuned on 150,000+ sentences from 7 comprehensive cardiology textbooks
- **Clinical Focus:** Captures procedural knowledge, specialized terminology, and clinical reasoning patterns
- **Superior Performance:** Outperforms MedTE, PubMedBERT, and BioBERT on cardiology-specific semantic tasks
- **Efficient Architecture:** LoRA-based fine-tuning (117 MB adapter) on top of Qwen3-Embedding-8B base model
- **Competitive Generalization:** Maintains strong performance on general biomedical benchmarks (BIOSSES, SciFact, NFCorpus)

---

## 🚀 Quick Start

### Using the Pre-trained Model

**Model:** [richardyoung/CardioEmbed](https://huggingface.co/richardyoung/CardioEmbed)

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and CardioEmbed adapter
base_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "richardyoung/CardioEmbed")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B", trust_remote_code=True)

# Generate embeddings for cardiology text
texts = [
    "Acute myocardial infarction with ST elevation in leads II, III, and aVF",
    "Transthoracic echocardiography shows reduced left ventricular ejection fraction"
]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state[:, -1, :]  # EOS pooling

print(f"Embedding shape: {embeddings.shape}")
# Output: torch.Size([2, 4096])

# Compute similarity
from torch.nn.functional import cosine_similarity
similarity = cosine_similarity(embeddings[0], embeddings[1], dim=0)
print(f"Cosine similarity: {similarity.item():.4f}")
```

---

## 📊 Performance

### Cardiology-Specific Retrieval
| Model | Acc@1 | Acc@5 | MRR | Mean Similarity |
|-------|-------|-------|-----|-----------------|
| **CardioEmbed (Ours)** | **99.60%** | **99.98%** | **0.9976** | **0.909** |
| Qwen3-8B-Base | 93.83% | 99.55% | 0.9665 | 0.886 |
| MedTE | 83.66% | 97.90% | 0.9029 | 0.859 |
| MedEmbed-base | 63.94% | 92.94% | 0.7810 | 0.792 |
| GTE-Base | 58.69% | 89.47% | 0.7305 | 0.783 |

### MTEB Medical Benchmarks
| Model | BIOSSES (Spearman) | SciFact (NDCG@10) | NFCorpus (NDCG@10) |
|-------|-------------------|------------------|-------------------|
| **CardioEmbed** | **0.77** | 0.61 | 0.26 |
| MedTE | 0.74 | **0.65** | **0.30** |
| Qwen3-8B-Base | 0.70 | 0.61 | 0.27 |

*CardioEmbed excels at cardiology-specific tasks while maintaining competitive performance on general biomedical benchmarks.*

---

## 🏗️ Repository Structure

```
CardioEmbed/
├── 01_merge_paraphrases.py           # Merge paraphrase data for training
├── 02_prepare_training_data.py       # Prepare training triplets
├── 03_train_embedding_model.py       # Main training script
├── 04_evaluate_model.py              # Evaluation on cardiology test set
├── 05_usage_example.py               # Example usage code
├── requirements.txt                  # Python dependencies
└── Final_Published_Paper/            # LaTeX source for paper
    ├── main.tex
    ├── bibliography/references.bib
    └── figures/
```

---

## 📚 Training Your Own Model

### Prerequisites

```bash
# Python 3.8+
python --version

# For training: NVIDIA GPU with 16GB+ VRAM
nvidia-smi

# Install dependencies
pip install -r requirements.txt
```

### Training Pipeline

```bash
# 1. Prepare your cardiology training data
python 02_prepare_training_data.py

# 2. (Optional) Generate paraphrases for data augmentation
python 01_merge_paraphrases.py

# 3. Train the embedding model
python 03_train_embedding_model.py

# 4. Evaluate on your test set
python 04_evaluate_model.py
```

### Training Data

**Corpus:** 7 comprehensive cardiology textbooks (legally purchased)
- Braunwald's Heart Disease (11th ed, 2018)
- ESC Textbook of Cardiovascular Imaging (3rd ed, 2021)
- Textbook of Cardiovascular Medicine (2nd ed)
- Echocardiography Review Guide (4th ed, 2019)
- Intraprocedural Imaging of Cardiovascular Interventions (2016)
- Additional specialized cardiology references

**Processing:**
- OCR: DeepSeek-OCR (3B parameter vision-language model)
- Sentence extraction and deduplication
- Final corpus: ~150,000 unique sentences
- Training triplets: 106,386 (anchor, positive, hard negative)

**Note:** Due to copyright restrictions, the processed training corpus cannot be publicly released. All source materials were legally acquired.

---

## 🔬 Training Configuration

### Model Architecture
- **Base:** Qwen3-Embedding-8B (8B parameters, 28 layers)
- **Fine-tuning:** LoRA (rank=16, alpha=32)
- **Embedding:** EOS token pooling
- **Dimensions:** 4096
- **Quantization:** INT8 (base model), FP32 (LoRA adapters)

### Training Hyperparameters
- **Loss:** InfoNCE with in-batch negatives (τ=0.05)
- **Batch size:** 128
- **Learning rate:** 2e-4 (AdamW)
- **Schedule:** 10% linear warmup + cosine annealing
- **Epochs:** 2
- **Hardware:** NVIDIA H100 PCIe (80GB VRAM)
- **Training time:** ~11 hours

---

## 💻 System Requirements

### For Training
- **GPU:** NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100, H100)
- **RAM:** 32GB+ recommended
- **Storage:** 50GB+ for model, data, and checkpoints
- **CUDA:** 11.8+ or 12.x

### For Inference
- **RAM:** 16GB+
- **GPU:** Optional (CPU inference supported, but slower)
- **Storage:** ~20GB (base model + adapter)

---

## 📜 Citation

If you use CardioEmbed in your research, please cite:

```bibtex
@article{young2025cardioembed,
  title={CardioEmbed: Domain-Specialized Text Embeddings for Clinical Cardiology},
  author={Young, Richard J. and Matthews, Alice M.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 🤝 Contributing

This is a research repository. For questions, issues, or collaboration inquiries:
- **Email:** ryoung@unlv.edu
- **Issues:** Use GitHub Issues for bug reports or feature requests

---

## 📄 License

**Apache License 2.0**

Copyright (c) 2025 Richard J. Young

See [LICENSE](LICENSE) for full details.

---

## 🙏 Acknowledgments

- **Base model:** Qwen Team (Alibaba Cloud) - Qwen3-Embedding-8B
- **OCR:** DeepSeek-AI - DeepSeek-OCR
- **Benchmarking:** MTEB (Massive Text Embedding Benchmark)
- **Computational resources:** NVIDIA H100 GPU infrastructure

---

## ⚠️ Important Notes

### Data Ethics
- All textbooks were legally purchased
- Training corpus cannot be shared due to copyright
- Model adapter weights released under Apache 2.0

### Medical AI Disclaimer
This model is intended for **research purposes only**. It should not be used for clinical decision-making without proper validation and oversight by qualified medical professionals.

---

**Last Updated:** November 2025
**Status:** Published Research Project
