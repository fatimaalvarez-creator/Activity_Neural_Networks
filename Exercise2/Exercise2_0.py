"""
Ejercicio 2 (Partes 1 y 2) – MLP Clasificación Binaria y de 4 Clases
=====================================================================
Parte 1: MLP para clasificación binaria (misterious_data_1.txt)
Parte 2: MLP para clasificación de 4 clases (misterious_data_4.txt)

Ambas partes usan validación cruzada de 5 folds.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Carga de datos ──────────────────────────────────────────────────────────
def load_data(path):
    raw = np.loadtxt(path)
    X = raw[:, 1:]
    y = raw[:, 0]
    return X, y

# Dataset binario (etiquetas 1, 2 → se quedan como están para sklearn)
X1, y1 = load_data('Exercise2\misterious_data_1.txt')
print(f"Dataset 1 (binario): {X1.shape[0]} muestras, {X1.shape[1]} características")
print(f"  Clases: {np.unique(y1)}")

# Dataset de 4 clases
X4, y4 = load_data('Exercise2\misterious_data_4.txt')
print(f"\nDataset 4 (4 clases): {X4.shape[0]} muestras, {X4.shape[1]} características")
print(f"  Clases: {np.unique(y4)}")


# ── Función de validación cruzada ──────────────────────────────────────────
def mlp_classification_cv(X, y, hidden_layers=(128, 64), k=5, seed=42):
    """
    Entrena un MLPClassifier con k-fold CV y devuelve:
      - errores por fold
      - precisión media
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    errors = []

    for fold, (tr, te) in enumerate(kf.split(X)):
        # Normalización: ajuste solo en train, aplicado en test
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=seed
        )
        clf.fit(Xtr, y[tr])
        err = 1 - clf.score(Xte, y[te])
        errors.append(err)
        print(f"  Fold {fold+1}: error = {err:.4f}  ({(1-err)*100:.1f}% acc)")

    return np.array(errors)


# ══════════════════════════════════════════════════════════════════════════
# PARTE 1 – Clasificación Binaria (Dataset 1)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Parte 1: MLP – Clasificación Binaria ===")
print(f"Arquitectura: {X1.shape[1]} → 128 → 64 → 1")
errors_bin = mlp_classification_cv(X1, y1, hidden_layers=(128, 64))

print(f"\nResultado binario:")
print(f"  Error promedio : {errors_bin.mean():.4f} ± {errors_bin.std():.4f}")
print(f"  Precisión media: {(1 - errors_bin.mean())*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# PARTE 2 – Clasificación de 4 Clases (Dataset 4)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Parte 2: MLP – Clasificación de 4 Clases ===")
print(f"Arquitectura: {X4.shape[1]} → 128 → 64 → 4 (softmax)")
errors_4c = mlp_classification_cv(X4, y4, hidden_layers=(128, 64))

print(f"\nResultado 4 clases:")
print(f"  Error promedio : {errors_4c.mean():.4f} ± {errors_4c.std():.4f}")
print(f"  Precisión media: {(1 - errors_4c.mean())*100:.1f}%")


# ── Gráfica comparativa ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Ejercicio 2 (Partes 1 y 2) – Error por Fold (Validación Cruzada)',
             fontsize=13, fontweight='bold')

datasets = [
    (errors_bin * 100, 'Clasificación Binaria\n(Dataset 1)', '#457b9d'),
    (errors_4c  * 100, 'Clasificación 4 Clases\n(Dataset 4)', '#e76f51'),
]

for ax, (vals, title, color) in zip(axes, datasets):
    folds = np.arange(1, len(vals) + 1)
    bars = ax.bar(folds, vals, color=color, alpha=0.85, edgecolor='white', linewidth=1.2)
    ax.axhline(vals.mean(), color='black', linestyle='--', linewidth=1.5,
               label=f'Media = {vals.mean():.1f}%')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vals.max() * 0.01,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Error de Clasificación (%)')
    ax.set_xticks(folds)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.25)

plt.tight_layout()
plt.savefig('ejercicio2_clasificacion.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfica guardada: ejercicio2_clasificacion.png")