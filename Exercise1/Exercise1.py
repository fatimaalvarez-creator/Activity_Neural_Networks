"""
Ejercicio 1 — Perceptrón de una neurona + SVM de una neurona
Dataset: misterious_data_1.txt (2 clases)
Variantes: SGD, Batch GD, Mini-batch GD
Evaluación: Cross-validación de 5 folds, 100 épocas
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────
data = np.loadtxt('Exercise1/misterious_data_1.txt', delimiter='\t')
X = data[:, 1:]
y = data[:, 0].astype(int)
y[y == 2] = -1   # etiquetas: 1 y -1

# ─────────────────────────────────────────────
# Funciones del Perceptrón
# ─────────────────────────────────────────────
def perceptron(x, w):
    """Predicción de una sola muestra."""
    return 1 if x @ w > 0 else -1

def perceptron_mult(X, w):
    """Predicción vectorizada."""
    return np.sign(X @ w)

def classification_error(X, y, w):
    return np.mean(perceptron_mult(X, w) != y)

# ── Perceptrón SGD ──────────────────────────
def train_perceptron_sgd(X_tr, y_tr, X_te, y_te, alpha=0.01, n_epochs=100):
    n, d = X_tr.shape
    w = np.zeros(d)
    errors = []
    for _ in range(n_epochs):
        idx = np.random.permutation(n)
        for i in idx:
            yp = perceptron(X_tr[i], w)
            if y_tr[i] != yp:
                w += alpha * y_tr[i] * X_tr[i]
        errors.append(classification_error(X_te, y_te, w))
    return errors

# ── Perceptrón Batch GD ──────────────────────
def train_perceptron_batch(X_tr, y_tr, X_te, y_te, alpha=0.01, n_epochs=100):
    n, d = X_tr.shape
    w = np.zeros(d)
    errors = []
    for _ in range(n_epochs):
        wrong = (perceptron_mult(X_tr, w) != y_tr)
        if wrong.any():
            w += alpha * (X_tr[wrong].T @ y_tr[wrong])
        errors.append(classification_error(X_te, y_te, w))
    return errors

# ── Perceptrón Mini-batch GD ─────────────────
def train_perceptron_minibatch(X_tr, y_tr, X_te, y_te, alpha=0.01,
                               n_epochs=100, batch_size=32):
    n, d = X_tr.shape
    w = np.zeros(d)
    errors = []
    for _ in range(n_epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            wrong = (perceptron_mult(X_tr[b], w) != y_tr[b])
            if wrong.any():
                w += alpha * (X_tr[b][wrong].T @ y_tr[b][wrong])
        errors.append(classification_error(X_te, y_te, w))
    return errors

# ─────────────────────────────────────────────
# Funciones del SVM de una neurona
# Regla: hinge loss + regularización L2
# ─────────────────────────────────────────────
def train_svm_sgd(X_tr, y_tr, X_te, y_te, alpha=0.01, lam=0.001, n_epochs=100):
    n, d = X_tr.shape
    w = np.zeros(d)
    errors = []
    for _ in range(n_epochs):
        idx = np.random.permutation(n)
        for i in idx:
            if y_tr[i] * (X_tr[i] @ w) < 1:
                w -= alpha * (-y_tr[i] * X_tr[i] + lam * w)
            else:
                w -= alpha * lam * w
        errors.append(np.mean(np.sign(X_te @ w) != y_te))
    return errors

def train_svm_batch(X_tr, y_tr, X_te, y_te, alpha=0.005, lam=0.001, n_epochs=100):
    n, d = X_tr.shape
    w = np.zeros(d)
    errors = []
    for _ in range(n_epochs):
        margins = y_tr * (X_tr @ w)
        violated = margins < 1
        grad = ((-X_tr[violated].T @ y_tr[violated]) / n + lam * w
                if violated.any() else lam * w)
        w -= alpha * grad
        errors.append(np.mean(np.sign(X_te @ w) != y_te))
    return errors

def train_svm_minibatch(X_tr, y_tr, X_te, y_te, alpha=0.005, lam=0.001,
                        n_epochs=100, batch_size=32):
    n, d = X_tr.shape
    w = np.zeros(d)
    errors = []
    for _ in range(n_epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            margins = y_tr[b] * (X_tr[b] @ w)
            violated = margins < 1
            grad = ((-X_tr[b][violated].T @ y_tr[b][violated]) / len(b) + lam * w
                    if violated.any() else lam * w)
            w -= alpha * grad
        errors.append(np.mean(np.sign(X_te @ w) != y_te))
    return errors

# ─────────────────────────────────────────────
# Cross-validación (5 folds, 100 épocas)
# ─────────────────────────────────────────────
from sklearn.model_selection import KFold

N_SPLITS = 5
N_EPOCHS = 100
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Acumuladores: shape (n_splits, n_epochs)
results = {
    'perc_sgd':      [],
    'perc_batch':    [],
    'perc_mini':     [],
    'svm_sgd':       [],
    'svm_batch':     [],
    'svm_mini':      [],
}

print(f"Entrenando con {N_SPLITS}-fold CV, {N_EPOCHS} épocas...")

for fold, (tr, te) in enumerate(kf.split(X), 1):
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    print(f"  Fold {fold}/{N_SPLITS}", end='\r')

    results['perc_sgd'].append(
        train_perceptron_sgd(X_tr, y_tr, X_te, y_te, n_epochs=N_EPOCHS))
    results['perc_batch'].append(
        train_perceptron_batch(X_tr, y_tr, X_te, y_te, n_epochs=N_EPOCHS))
    results['perc_mini'].append(
        train_perceptron_minibatch(X_tr, y_tr, X_te, y_te, n_epochs=N_EPOCHS))
    results['svm_sgd'].append(
        train_svm_sgd(X_tr, y_tr, X_te, y_te, n_epochs=N_EPOCHS))
    results['svm_batch'].append(
        train_svm_batch(X_tr, y_tr, X_te, y_te, n_epochs=N_EPOCHS))
    results['svm_mini'].append(
        train_svm_minibatch(X_tr, y_tr, X_te, y_te, n_epochs=N_EPOCHS))

# Promedio sobre folds
avg = {k: np.mean(v, axis=0) for k, v in results.items()}

# ─────────────────────────────────────────────
# Resultados en consola
# ─────────────────────────────────────────────
print("\n=== Error de clasificación promedio (época 100) ===")
print(f"  Perceptrón SGD       : {avg['perc_sgd'][-1]:.4f}")
print(f"  Perceptrón Batch GD  : {avg['perc_batch'][-1]:.4f}")
print(f"  Perceptrón Mini-batch: {avg['perc_mini'][-1]:.4f}")
print(f"  SVM SGD              : {avg['svm_sgd'][-1]:.4f}")
print(f"  SVM Batch GD         : {avg['svm_batch'][-1]:.4f}")
print(f"  SVM Mini-batch       : {avg['svm_mini'][-1]:.4f}")

# ─────────────────────────────────────────────
# Gráfica: Error promedio vs. Época
# ─────────────────────────────────────────────
epochs = range(1, N_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Perceptrón
axes[0].plot(epochs, avg['perc_sgd'],   label='SGD',        linewidth=2)
axes[0].plot(epochs, avg['perc_batch'], label='Batch GD',   linewidth=2)
axes[0].plot(epochs, avg['perc_mini'],  label='Mini-batch', linewidth=2)
axes[0].set_title('Perceptrón: Error vs. Época\n(promedio 5-fold CV)', fontsize=13)
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Error de Clasificación')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# SVM
axes[1].plot(epochs, avg['svm_sgd'],   label='SGD',        linewidth=2)
axes[1].plot(epochs, avg['svm_batch'], label='Batch GD',   linewidth=2)
axes[1].plot(epochs, avg['svm_mini'],  label='Mini-batch', linewidth=2)
axes[1].set_title('SVM (1 neurona): Error vs. Época\n(promedio 5-fold CV)', fontsize=13)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Error de Clasificación')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ejercicio1_error_vs_epoca.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfica guardada: ejercicio1_error_vs_epoca.png")