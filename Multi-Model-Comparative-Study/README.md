# CardioEmbed: Final Published Paper

**Clean, ready-to-upload folder for Overleaf submission**

---

## 📁 Contents

```
Final_Published_Paper/
├── main.tex                    # Main LaTeX document
├── bibliography/
│   └── references.bib          # All 10 references
├── figures/                    # All 11 figures (PDF format, 300 DPI)
│   ├── figure1_model_comparison.pdf
│   ├── figure2_mteb_benchmarks.pdf
│   ├── figure3_accuracy_at_k.pdf
│   ├── figure4_depth_vs_breadth.pdf
│   ├── figure5_corpus_composition.pdf
│   ├── figure6_metric_comparison.pdf
│   ├── figure7_mteb_heatmap.pdf          [IN PAPER]
│   ├── figure8_specialization_tradeoff.pdf
│   ├── figure9_improvement_waterfall.pdf  [IN PAPER]
│   ├── figure10_radar_performance.pdf     [IN PAPER]
│   └── figure11_mrr_comparison.pdf        [IN PAPER]
└── tables/                     # CSV data tables (5 tables)
    ├── table1_model_comparison.csv
    ├── table2_mteb_results.csv
    ├── table3_training_config.csv
    ├── table4_corpus_statistics.csv
    └── table5_retrieval_at_k.csv
```

---

## 📄 Paper Details

**Title:** CardioEmbed: Domain-Specialized Text Embeddings for Clinical Cardiology

**Authors:**
- Richard J. Young¹* (University of Nevada Las Vegas, Department of Neuroscience)
- Alice M. Matthews² (Concorde Career Colleges, Department of Cardiovascular and Medical Diagnostic Sonography)

**Corresponding:** ryoung@unlv.edu

---

## 📊 Paper Statistics

- **Figures:** 6 figures in paper (11 total available)
- **Tables:** 2 tables in paper
- **References:** 10 citations
- **Length:** ~214 lines LaTeX
- **Sections:** Abstract, Introduction, Methods, Results, Discussion, Conclusion

---

## 🎯 Key Results

- **99.60% Accuracy@1** on cardiology semantic retrieval
- **+15.94% improvement** over MedTE (SOTA medical embedding)
- **0.9976 MRR** (near-perfect ranking)
- **MTEB benchmarks:** BIOSSES 0.77, SciFact 0.61

---

## 🚀 Upload to Overleaf

### Option 1: Drag & Drop
1. Create new Overleaf project
2. Upload entire `Final_Published_Paper/` folder
3. Set `main.tex` as main document
4. Compile with pdflatex

### Option 2: Git Integration
```bash
cd Final_Published_Paper
git init
git add .
git commit -m "Initial commit: CardioEmbed paper"
# Connect to Overleaf git repository
```

### Compilation Settings:
- **Compiler:** pdfLaTeX
- **Main document:** main.tex
- **TeX Live version:** 2024 or later

---

## 📝 Figures in Paper

The paper currently uses **6 of 11 available figures**:

### **Currently Used:**
1. Figure 1: Model Comparison (Results 3.2)
2. Figure 3: Accuracy@K Curves (Results 3.2)
3. Figure 7: MTEB Heatmap (Results 3.3)
4. Figure 9: Improvement Waterfall (Discussion 4)
5. Figure 10: Radar Performance (Discussion 4)
6. Figure 11: MRR Comparison (Results 3.2)

### **Available for Supplementary:**
- Figure 2: MTEB Benchmarks
- Figure 4: Depth vs. Breadth
- Figure 5: Corpus Composition
- Figure 6: Metric Comparison
- Figure 8: Specialization Trade-off

---

## 🔧 Local Compilation

```bash
cd Final_Published_Paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Output:** `main.pdf`

---

## ✅ Verification Checklist

Before submission:
- [x] All figures render correctly
- [x] All references cited and formatted
- [x] Author affiliations correct
- [x] Abstract under word limit
- [x] All equations numbered
- [x] All tables formatted
- [x] Bibliography complete
- [x] No compilation errors

---

## 📋 Version History

**Version 1.0** (November 10, 2025)
- Complete paper with 6 figures
- 10 references
- All results included
- Ready for submission

---

## 📞 Contact

**Primary Author:** Richard J. Young
**Email:** ryoung@unlv.edu
**Institution:** University of Nevada Las Vegas

**Co-Author:** Alice M. Matthews
**Institution:** Concorde Career Colleges

---

## 🎓 Citation

```bibtex
@article{young2025cardioembed,
  title={CardioEmbed: Domain-Specialized Text Embeddings for Clinical Cardiology},
  author={Young, Richard J and Matthews, Alice M},
  journal={[To be determined]},
  year={2025}
}
```

---

**This folder contains the complete, final version ready for journal submission and Overleaf upload.**
