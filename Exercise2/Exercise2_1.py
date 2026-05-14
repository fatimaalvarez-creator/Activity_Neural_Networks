"""
Ejercicio 2 (Parte 3) – MLP Regresión: Dataset Diabetes
=========================================================
Ajusta un MLP de regresión al dataset de diabetes de sklearn.
Evalúa el Error Cuadrático Medio (MSE) con validación cruzada de 5 folds.

Dataset: sklearn.datasets.load_diabetes()
  - 442 muestras, 10 características numéricas normalizadas
  - Variable objetivo: progresión de la enfermedad (continua)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── Carga del dataset ───────────────────────────────────────────────────────
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print("=== Dataset: Diabetes (sklearn) ===")
print(f"  Muestras    : {X.shape[0]}")
print(f"  Características: {X.shape[1]}")
print(f"  Target (y)  : min={y.min():.1f}, max={y.max():.1f}, media={y.mean():.1f}")
print(f"  Características: {list(diabetes.feature_names)}")


# ── Validación cruzada con MLP Regressor ────────────────────────────────────
def mlp_regression_cv(X, y, hidden_layers=(128, 64, 32), k=5, seed=42):
    """
    Entrena MLPRegressor con k-fold CV.
    Devuelve array de MSE por fold.
    """
    kf  = KFold(n_splits=k, shuffle=True, random_state=seed)
    mse_folds = []

    for fold, (tr, te) in enumerate(kf.split(X)):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])

        reg = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=seed
        )
        reg.fit(Xtr, y[tr])
        preds = reg.predict(Xte)
        mse   = mean_squared_error(y[te], preds)
        rmse  = np.sqrt(mse)
        mse_folds.append(mse)
        print(f"  Fold {fold+1}: MSE = {mse:.2f}  |  RMSE = {rmse:.2f}")

    return np.array(mse_folds)


print("\n=== MLP Regresión – 5-Fold CV ===")
print(f"Arquitectura: 10 → 128 → 64 → 32 → 1")
mse_folds = mlp_regression_cv(X, y, hidden_layers=(128, 64, 32))

print(f"\nResultados finales:")
print(f"  MSE  promedio : {mse_folds.mean():.2f} ± {mse_folds.std():.2f}")
print(f"  RMSE promedio : {np.sqrt(mse_folds).mean():.2f} ± {np.sqrt(mse_folds).std():.2f}")


# ── Gráfica ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Ejercicio 2 (Parte 3) – MLP Regresión: Dataset Diabetes',
             fontsize=13, fontweight='bold')

folds = np.arange(1, len(mse_folds) + 1)
rmse_folds = np.sqrt(mse_folds)

# Gráfica 1: MSE por fold
axes[0].bar(folds, mse_folds, color='#2a9d8f', alpha=0.85, edgecolor='white', linewidth=1.2)
axes[0].axhline(mse_folds.mean(), color='black', linestyle='--', linewidth=1.5,
                label=f'MSE medio = {mse_folds.mean():.1f}')
for i, v in enumerate(mse_folds):
    axes[0].text(folds[i], v + mse_folds.max()*0.01,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=9)
axes[0].set_title('MSE por Fold', fontweight='bold')
axes[0].set_xlabel('Fold'); axes[0].set_ylabel('MSE')
axes[0].set_xticks(folds); axes[0].legend(fontsize=9); axes[0].grid(axis='y', alpha=0.3)

# Gráfica 2: RMSE por fold
axes[1].bar(folds, rmse_folds, color='#e76f51', alpha=0.85, edgecolor='white', linewidth=1.2)
axes[1].axhline(rmse_folds.mean(), color='black', linestyle='--', linewidth=1.5,
                label=f'RMSE medio = {rmse_folds.mean():.1f}')
for i, v in enumerate(rmse_folds):
    axes[1].text(folds[i], v + rmse_folds.max()*0.01,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=9)
axes[1].set_title('RMSE por Fold', fontweight='bold')
axes[1].set_xlabel('Fold'); axes[1].set_ylabel('RMSE (unidades de progresión)')
axes[1].set_xticks(folds); axes[1].legend(fontsize=9); axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ejercicio2_regresion_diabetes.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfica guardada: ejercicio2_regresion_diabetes.png")