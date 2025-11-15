

```markdown
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R05VM8Rg)

# IIT-Madras-DA2401-Machine-Learning-Lab-End-Semester-Project

## Purpose of this Template

This repository contains a **complete from-scratch implementation** of a multi-class digit classification system for the **MNIST dataset**, built **without using scikit-learn** or any high-level ML frameworks. All algorithms (PCA, k-NN, Logistic Regression, Bagging, etc.) are implemented using only **NumPy and standard Python libraries**.

The final champion model — **k-NN (k=5) on 40 PCA components** — achieves a **validation F1-score of 0.9583** with training time under **0.3 seconds**, well within the 5-minute limit.

All code, analysis, visualizations, and the final report are included for full reproducibility.

---

## Repository Structure

```
END-SEMESTER-PROJECT-RINKESH-1612/
│
├── main.py                  # Entry point: runs full pipeline (load → PCA → k-NN → eval)
├── algorithms.py            # Core ML implementations: PCA, KNearestNeighbors, LogisticRegression, Bagging
├── analysis.py              # Post-hoc analysis: generates 5 report figures + observations
├── report/
|── report_figures/      # All PNGs: confusion matrix, F1 scores, PCA   variance, 2D proj, misclassified
│       ├── confusion_matrix.png
│       ├── per_class_f1.png
│       ├── pca_variance_explained.png
│       ├── class_separation_2d.png
│       └── misclassified_examples.png
|
├── MNIST_train.csv
└── MNIST_validation.csv
├── ml_lab_end_sem_report.pdf # Final LaTeX report with in-depth analysis
└── README.md
```

> **Note**: All visualizations are generated **on-the-fly** by `analysis.py` and saved to `report/report_figures/`.

---

## Installation & Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/END-SEMESTER-PROJECT-RINKESH-1612.git
cd END-SEMESTER-PROJECT-RINKESH-1612

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install numpy pandas matplotlib seaborn
```

> **No scikit-learn used** — all algorithms are built from scratch.

---

## Running the Code

All experiments are **reproducible** and run from the command line.

### A. Run the Full Champion Model (Recommended for Grading)

```bash
python main.py
```

**Output**:
- Loads data
- Applies PCA (40 components)
- Trains k-NN (k=5)
- Prints: **F1-Score: 0.9583**, **Accuracy: 0.9584**, **Training Time: ~0.21s**

---

### B. Generate Report Figures & Analysis

```bash
python analysis.py
```

**Output**:
- Generates **5 high-quality PNGs** in `report/report_figures/`
- Prints **5 key observations** (used in the report)
- Saves: confusion matrix, per-class F1, PCA variance, 2D projection, misclassified examples

---

### C. View Final Report

Open `ml_lab_end_sem_report.pdf` — includes:
- Full methodology
- Hyperparameter tuning results
- In-depth model analysis with **5 figures**
- 5 data-driven observations
- Conclusion & system recommendation

---

## Key Results

| Model | Features | F1-Score | Time |
|------|----------|----------|------|
| **k-NN (k=5)** | **PCA (40)** | **0.9583** | **0.21s** |
| k-NN (k=3) | PCA (40) | 0.9565 | 0.20s |
| Bagging (100 trees) | Raw pixels | 0.8993 | 48s |
| Logistic Regression | HOG | 0.9099 | 12s |

> **PCA + k-NN** is the clear winner: **fast, accurate, elegant**.

---

## Authors

**Rinkesh Patel DA24B017**, IIT Madras (2025–26)

---

## Best Practices Followed

- **Modular code**: `algorithms.py`, `main.py`, `analysis.py` — clean separation
- **Meaningful commits**: Incremental development (data → PCA → k-NN → tuning → analysis)
- **Well-commented**: Docstrings, inline comments, and print statements
- **Reproducible**: Fixed random seeds, deterministic PCA, no external dependencies
- **No collaboration**: Fully independent work

---

> **TAs**: Please evaluate using `main.py` and `analysis.py`. All plots and the final report are included.
```


