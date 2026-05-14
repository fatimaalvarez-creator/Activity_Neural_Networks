"""
Ejercicio 1 – Perceptrón de una neurona y SVM de una neurona
============================================================
Implementa:
  - Perceptrón (sigmoide / pérdida BCE)
  - SVM (pérdida hinge)
Ambos entrenados con SGD, Batch GD y Mini-Batch GD.
Evaluación con validación cruzada de 5 folds, 150 épocas.
Dataset: misterious_data_1.txt (primera columna = etiqueta)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Carga de datos ─────────────────────────────────────────────────────────
def load_binary_data(path):
    raw = np.loadtxt(path)
    X = raw[:, 1:]
    y = raw[:, 0]
    classes = np.unique(y)
    y = np.where(y == classes[0], -1, 1).astype(int)
    return X, y

X, y = load_binary_data('Exercise1\misterious_data_1.txt')
print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
print(f"Etiquetas: {np.unique(y)}")

# ── Activación sigmoide ────────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# ══════════════════════════════════════════════════════════════════════════
# PERCEPTRÓN DE UNA NEURONA  (sigmoide / pérdida BCE)
# ══════════════════════════════════════════════════════════════════════════
class Perceptron:
    """
    Perceptrón de una neurona con activación sigmoide.
    Modos: 'sgd', 'batch', 'minibatch'
    """
    def __init__(self, lr=0.05, epochs=150, mode='sgd', batch_size=32, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed

    def _init_weights(self, n):
        rng = np.random.RandomState(self.seed)
        self.w = rng.randn(n) * 0.01
        self.b = 0.0

    def _forward(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return np.where(self._forward(X) >= 0.5, 1, -1)

    def _update(self, X, y_bin):
        """Una pasada de actualización de pesos según el modo."""
        n = len(X)
        if self.mode == 'sgd':
            for i in np.random.permutation(n):
                xi, yi = X[i:i+1], y_bin[i:i+1]
                grad = self._forward(xi) - yi
                self.w -= self.lr * xi[0] * grad[0]
                self.b -= self.lr * grad[0]

        elif self.mode == 'batch':
            pred = self._forward(X)
            grad = pred - y_bin
            self.w -= self.lr * (X.T @ grad) / n
            self.b -= self.lr * grad.mean()

        elif self.mode == 'minibatch':
            for start in range(0, n, self.batch_size):
                idx = np.random.permutation(n)[start:start+self.batch_size]
                xi, yi = X[idx], y_bin[idx]
                grad = self._forward(xi) - yi
                self.w -= self.lr * (xi.T @ grad) / len(idx)
                self.b -= self.lr * grad.mean()

    def cross_validate(self, X, y, k=5):
        """Devuelve (épocas, error_promedio_por_época)."""
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_errors = np.zeros((k, self.epochs))

        for fold, (tr, te) in enumerate(kf.split(X)):
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr])
            Xte = scaler.transform(X[te])
            y_bin_tr = (y[tr] == 1).astype(float)

            self._init_weights(Xtr.shape[1])
            for epoch in range(self.epochs):
                self._update(Xtr, y_bin_tr)
                fold_errors[fold, epoch] = np.mean(self.predict(Xte) != y[te])

        return np.arange(1, self.epochs + 1), fold_errors.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════
# SVM DE UNA NEURONA  (pérdida hinge + regularización L2)
# ══════════════════════════════════════════════════════════════════════════
class SingleNeuronSVM:
    """
    SVM de una neurona entrenada con sub-gradiente de la pérdida hinge.
    Modos: 'sgd', 'batch', 'minibatch'
    """
    def __init__(self, lr=0.01, C=1.0, epochs=150, mode='sgd', batch_size=32, seed=42):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed

    def _init_weights(self, n):
        rng = np.random.RandomState(self.seed)
        self.w = rng.randn(n) * 0.01
        self.b = 0.0

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def _hinge_grad(self, X, y):
        """Sub-gradiente de hinge loss con regularización L2."""
        margins = y * (X @ self.w + self.b)
        mask = (margins < 1).astype(float)
        gw = self.w - self.C * (mask[:, None] * y[:, None] * X).mean(axis=0)
        gb = -self.C * (mask * y).mean()
        return gw, gb

    def _update(self, X, y):
        n = len(X)
        if self.mode == 'sgd':
            for i in np.random.permutation(n):
                gw, gb = self._hinge_grad(X[i:i+1], y[i:i+1])
                self.w -= self.lr * gw
                self.b -= self.lr * gb

        elif self.mode == 'batch':
            gw, gb = self._hinge_grad(X, y)
            self.w -= self.lr * gw
            self.b -= self.lr * gb

        elif self.mode == 'minibatch':
            for start in range(0, n, self.batch_size):
                idx = np.random.permutation(n)[start:start+self.batch_size]
                gw, gb = self._hinge_grad(X[idx], y[idx])
                self.w -= self.lr * gw
                self.b -= self.lr * gb

    def cross_validate(self, X, y, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_errors = np.zeros((k, self.epochs))

        for fold, (tr, te) in enumerate(kf.split(X)):
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr])
            Xte = scaler.transform(X[te])

            self._init_weights(Xtr.shape[1])
            for epoch in range(self.epochs):
                self._update(Xtr, y[tr])
                fold_errors[fold, epoch] = np.mean(self.predict(Xte) != y[te])

        return np.arange(1, self.epochs + 1), fold_errors.mean(axis=0)


# ── Entrenamiento y gráfica ────────────────────────────────────────────────
EPOCHS = 150
MODES  = ['sgd', 'batch', 'minibatch']
LABELS = {'sgd': 'Stochastic GD', 'batch': 'Batch GD', 'minibatch': 'Mini-Batch GD'}
COLORS = {'sgd': '#e63946', 'batch': '#2a9d8f', 'minibatch': '#e9c46a'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Ejercicio 1 – Error de Clasificación vs Época', fontsize=14, fontweight='bold')

# — Perceptrón —
print("\n=== Perceptrón (BCE) ===")
for m in MODES:
    model = Perceptron(lr=0.05, epochs=EPOCHS, mode=m)
    ep, err = model.cross_validate(X, y)
    print(f"  {LABELS[m]:20s}  Error final: {err[-1]:.4f}  ({(1-err[-1])*100:.1f}% acc)")
    axes[0].plot(ep, err, label=LABELS[m], color=COLORS[m], linewidth=2)

axes[0].set_title('Perceptrón (Sigmoide / BCE)', fontweight='bold')
axes[0].set_xlabel('Época'); axes[0].set_ylabel('Error de Clasificación Promedio')
axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 1)

# — SVM —
print("\n=== SVM (Hinge) ===")
for m in MODES:
    model = SingleNeuronSVM(lr=0.01, C=1.0, epochs=EPOCHS, mode=m)
    ep, err = model.cross_validate(X, y)
    print(f"  {LABELS[m]:20s}  Error final: {err[-1]:.4f}  ({(1-err[-1])*100:.1f}% acc)")
    axes[1].plot(ep, err, label=LABELS[m], color=COLORS[m], linewidth=2)

axes[1].set_title('SVM (Pérdida Hinge)', fontweight='bold')
axes[1].set_xlabel('Época'); axes[1].set_ylabel('Error de Clasificación Promedio')
axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('ejercicio1_resultados.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfica guardada: ejercicio1_resultados.png")