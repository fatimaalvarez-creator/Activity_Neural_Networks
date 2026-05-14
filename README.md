# 🤖 Machine Learning — Perceptron, SVM & MLP

> **Course activity:** Implement and evaluate single-neuron and multilayer neural network models using stochastic, batch, and mini-batch gradient descent, validated with cross-validation techniques.

---

## 📋 Overview

This repository contains two exercises that explore the full workflow of a supervised machine learning project — from manually implementing learning rules to fitting MLPs for classification and regression. Everything is implemented in Python with **NumPy**, **scikit-learn**, and **Matplotlib**.

---

## 🗂️ Repository Structure

```
📦 ACTIVITY_NEURAL_NETWORKS
├── 📁 Exercise1/
│   ├── 📄 Exercise1.py              ← Single-neuron Perceptron & SVM
│   ├── 🖼️ exercise1.png             ← Convergence plots
│   └── 📄 misterious_data_1.txt
├── 📁 Exercise2/
│   ├── 📄 Exercise2_0.py            ← MLP binary & 4-class classification
│   ├── 📄 Exercise2_1.py            ← MLP regression (diabetes dataset)
│   ├── 🖼️ exercise2_0.png           ← Per-fold error plots (classification)
│   ├── 🖼️ exercise2_1.png           ← Per-fold MSE/RMSE plots (regression)
│   ├── 📄 misterious_data_1.txt
│   └── 📄 misterious_data_4.txt
├── 📄 Neural_Networks.ipynb
└── 📄 README.md
```

---

## 📚 Exercises Summary

### Exercise 1 — Single-Neuron Models (Perceptron & SVM)
**Dataset:** `misterious_data_1.txt` (528 samples, 153 features, 2 balanced classes)

Two single-neuron models were implemented from scratch and evaluated with **5-fold cross-validation** over **150 epochs**, comparing three gradient descent variants:

#### Perceptron (Sigmoid Activation / BCE Loss)

| Method | Final Error | Accuracy |
|--------|-------------|----------|
| Stochastic GD | 0.2652 | 73.5% |
| 🥇 Batch GD | 0.2064 | **79.4%** |
| Mini-Batch GD | 0.2217 | 77.8% |

#### Single-Neuron SVM (Hinge Loss + L2 Regularization)

| Method | Final Error | Accuracy |
|--------|-------------|----------|
| Stochastic GD | 0.2557 | 74.4% |
| 🥇 Batch GD | **0.1932** | **80.7%** |
| 🥇 Mini-Batch GD | **0.1931** | **80.7%** |

**Key finding:** Batch GD converges most smoothly and achieves the lowest final error in both models. The SVM learning rule (hinge loss) outperforms the sigmoid perceptron under Batch and Mini-Batch regimes, confirming the theoretical advantage of margin maximization. SGD converges faster per sample but produces noisier error curves.

---

### Exercise 2 — Multilayer Perceptron (MLP)

#### Part 1 — Binary Classification
**Dataset:** `misterious_data_1.txt` — Architecture: `153 → 128 → 64 → 1`

| Fold | Error (%) | Accuracy (%) |
|------|-----------|--------------|
| 1 | ~22.6 | ~77.4 |
| 2 | ~19.8 | ~80.2 |
| 3 | ~22.6 | ~77.4 |
| 4 | ~27.4 | ~72.6 |
| 5 | ~20.8 | ~79.2 |
| **Mean ± SD** | **22.35 ± 3.02** | **77.65%** |

#### Part 2 — 4-Class Classification
**Dataset:** `misterious_data_4.txt` (259 samples, 648 features, 4 classes) — Architecture: `648 → 128 → 64 → 4`

| Metric | Value |
|--------|-------|
| CV Error (mean) | 10.40% |
| CV Error (SD) | ± 3.54% |
| 🥇 Accuracy | **89.60%** |

#### Part 3 — Regression (Diabetes Dataset)
**Dataset:** `sklearn.datasets.load_diabetes()` (442 samples, 10 features) — Architecture: `10 → 128 → 64 → 32 → 1`

| Metric | Value |
|--------|-------|
| MSE (mean) | 3267.20 |
| MSE (SD) | ± 300.91 |
| RMSE | **57.16** |

**Key finding:** The 4-class MLP reaches 89.6% accuracy despite the high dimensionality and small sample size. The diabetes regression RMSE of ~57 is competitive against the linear baseline (~53) and could improve further with hyperparameter tuning.

---

## ⚙️ Setup & Requirements

### Install dependencies

```bash
pip install scikit-learn matplotlib numpy
```

### Run the scripts

```bash
# Exercise 1: Single-neuron Perceptron & SVM
cd Exercise1
python Exercise1.py

# Exercise 2: Binary & 4-class classification
cd Exercise2
python Exercise2_0.py

# Exercise 2: Diabetes regression
cd Exercise2
python Exercise2_1.py
```

### Required dataset files
The datasets are already included in each folder:
- `Exercise1/misterious_data_1.txt`
- `Exercise2/misterious_data_1.txt`
- `Exercise2/misterious_data_4.txt`

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3 | Programming language |
| NumPy | Manual model implementation |
| scikit-learn | MLP, cross-validation, metrics |
| Matplotlib | Results visualization |

---

## 📌 Key Concepts Covered

- **Gradient descent** in three variants: SGD, Batch GD, and Mini-Batch GD
- **Perceptron learning rule** with sigmoid activation and BCE loss
- **SVM learning rule** with hinge loss and L2 regularization (C=1.0)
- **5-fold cross-validation** with per-fold normalization (avoids data leakage)
- **MLPs with ReLU + Adam** for binary classification, multi-class classification, and regression
- **Convergence curves** to compare the behavior of each GD variant

---

*Implemented with Python 3 · NumPy · scikit-learn · Matplotlib*