# 🤖 Machine Learning — Perceptron, SVM & MLP

> **Course activity:** Implement and evaluate single-neuron and multilayer neural network models using stochastic, batch, and mini-batch gradient descent, validated with cross-validation techniques.

---

## 📋 Overview

This repository contains two exercises that explore the full workflow of a supervised machine learning project — from manually implementing learning rules to fitting MLPs for classification and regression. Exercise 1 is implemented from scratch with **NumPy** and **scikit-learn**. Exercise 2 uses **Keras (TensorFlow)** as required.

---

## 🗂️ Repository Structure

```
📦 ACTIVITY_NEURAL_NETWORKS
├── 📁 Exercise1/
│   ├── 📄 Exercise1.py                  ← Single-neuron Perceptron & SVM
│   ├── 🖼️ Exercise1.png                 ← Convergence plots
│   └── 📄 misterious_data_1.txt
├── 📁 Exercise2/
│   ├── 📄 Exercise2_1.py                ← MLP binary classification (Keras)
│   ├── 📄 Exercise2_2.py                ← MLP 4-class classification (Keras)
│   ├── 📄 Exercise2_3.py                ← MLP regression — Parkinson's (Keras)
│   ├── 🖼️ Exercise2_1.png               ← Per-fold accuracy plots (binary)
│   ├── 🖼️ Exercise2_2.png               ← Per-fold accuracy plots (4-class)
│   ├── 🖼️ Exercise2_3.png               ← Per-fold MSE/RMSE plots (regression)
│   ├── 📄 misterious_data_1.txt
│   ├── 📄 misterious_data_4.txt
│   └── 📄 parkinsons_updrs.data
├── 📓 Neural_Networks.ipynb
└── 📄 README.md
```

---

## 📚 Exercises Summary

### Exercise 1 — Single-Neuron Models (Perceptron & SVM)
**Dataset:** `misterious_data_1.txt` (528 samples, 153 features, 2 balanced classes)

Two single-neuron models were implemented from scratch and evaluated with **5-fold cross-validation** over **100 epochs**, comparing three gradient descent variants:

#### Perceptron (classical learning rule)

| Method | Final Error | Accuracy |
|--------|-------------|----------|
| Stochastic GD | 0.4318 | 56.8% |
| Mini-Batch GD | 0.1795 | 82.1% |
| 🥇 Batch GD | **0.2121** | **78.8%** |

#### Single-Neuron SVM (Hinge Loss + L2 Regularization)

| Method | Final Error | Accuracy |
|--------|-------------|----------|
| 🥇 Stochastic GD | **0.1405** | **85.9%** |
| Mini-Batch GD | 0.1557 | 84.4% |
| Batch GD | 0.1989 | 80.1% |

**Key finding:** The SVM learning rule (hinge loss + L2 regularization) outperforms the classic perceptron across all gradient descent variants, confirming the theoretical advantage of margin maximization. SGD achieves the best result for SVM due to its frequent updates, while Mini-Batch GD is the best variant for the standard perceptron.

---

### Exercise 2 — Multilayer Perceptron with Keras

> ⚠️ All neural networks in this exercise are implemented in **Keras (TensorFlow)**.

#### Exercise 2.1 — Binary Classification
**Dataset:** `misterious_data_1.txt` — Architecture: `153 → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)`

| Fold | Accuracy |
|------|----------|
| 1 | 79.25% |
| 2 | 82.08% |
| 3 | 79.25% |
| 4 | 78.10% |
| 5 | 82.86% |
| **Mean ± SD** | **80.30% ± 1.83%** |

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Class 1 | 0.80 | 0.81 | 0.80 |
| Class 2 | 0.81 | 0.80 | 0.80 |

#### Exercise 2.2 — 4-Class Classification
**Dataset:** `misterious_data_4.txt` (259 samples, 648 features, 4 classes)  
**Architecture:** `648 → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dense(4, Softmax)`

| Fold | Accuracy |
|------|----------|
| 1 | 92.31% |
| 2 | 86.54% |
| 3 | 94.23% |
| 4 | 100.00% |
| 5 | 84.31% |
| **Mean ± SD** | **91.48% ± 5.60%** |

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Class 1 | 0.98 | 1.00 | 0.99 |
| Class 2 | 1.00 | 1.00 | 1.00 |
| Class 3 | 0.80 | 0.91 | 0.85 |
| Class 4 | 0.89 | 0.75 | 0.82 |

#### Exercise 2.3 — Regression (Parkinson's Telemonitoring)
**Dataset:** `parkinsons_updrs.data` — [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) (5,875 samples, 19 features)  
**Architecture:** `19 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(2, linear)`  
**Targets:** `motor_UPDRS` and `total_UPDRS` (multi-output regression)

| Target | MSE (mean ± SD) | RMSE |
|--------|-----------------|------|
| motor_UPDRS | 10.75 ± 1.12 | **3.28** |
| total_UPDRS | 18.46 ± 2.37 | **4.30** |

**Key finding:** The 4-class MLP achieves 91.5% accuracy despite the high dimensionality and small sample size. The Parkinson's regression model predicts both UPDRS scores simultaneously with competitive RMSE values, demonstrating the viability of MLP for multi-output clinical regression.

---

## ⚙️ Setup & Requirements

### Install dependencies

```bash
pip install tensorflow scikit-learn matplotlib numpy pandas seaborn
```

### Run the scripts

```bash
# Exercise 1: Single-neuron Perceptron & SVM
cd Exercise1
python Exercise1.py

# Exercise 2.1: Binary classification (Keras)
cd Exercise2
python Exercise2_1.py

# Exercise 2.2: 4-class classification (Keras)
python Exercise2_2.py

# Exercise 2.3: Parkinson's regression (Keras)
python Exercise2_3.py
```

### Required dataset files

| File | Location | Used in |
|------|----------|---------|
| `misterious_data_1.txt` | `Exercise1/`, `Exercise2/` | Ex. 1, Ex. 2.1 |
| `misterious_data_4.txt` | `Exercise2/` | Ex. 2.2 |
| `parkinsons_updrs.data` | `Exercise2/` | Ex. 2.3 |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.11 | Programming language |
| NumPy | Manual model implementation (Exercise 1) |
| TensorFlow / Keras | MLP models (Exercise 2) |
| scikit-learn | Cross-validation, metrics, preprocessing |
| Matplotlib / Seaborn | Results visualization |
| Pandas | Data loading (Parkinson's dataset) |

---

## 📌 Key Concepts Covered

- **Gradient descent** in three variants: SGD, Batch GD, and Mini-Batch GD
- **Perceptron learning rule** implemented from scratch with NumPy
- **SVM learning rule** with hinge loss and L2 regularization
- **5-fold cross-validation** with per-fold normalization (avoids data leakage)
- **MLPs with ReLU + Adam** for binary classification, multi-class classification, and regression
- **Multi-output regression** predicting two clinical targets simultaneously
- **Convergence curves** to compare the behavior of each GD variant

---

*Implemented with Python 3.11 · NumPy · TensorFlow/Keras · scikit-learn · Matplotlib*