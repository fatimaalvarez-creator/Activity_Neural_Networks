"""
Ejercicio 2.3 — MLP de Regresión: Parkinson's Telemonitoring (Keras)
Dataset: parkinsons_updrs.data (UCI Machine Learning Repository)
Objetivo: Predecir motor_UPDRS y total_UPDRS simultáneamente
Evaluación: Cross-validación de 5 folds (MSE y RMSE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────
# Carga y preparación de datos
# ─────────────────────────────────────────────
park = pd.read_csv('Exercise2/parkinsons_updrs.data')

TARGET_COLS  = ['motor_UPDRS', 'total_UPDRS']
DROP_COLS    = ['subject#'] + TARGET_COLS
FEATURE_COLS = [c for c in park.columns if c not in DROP_COLS]

X = park[FEATURE_COLS].values.astype(np.float32)
y = park[TARGET_COLS].values.astype(np.float32)

print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
print(f"Variables objetivo: {TARGET_COLS}")
print(f"\nEstadísticas de las variables objetivo:")
print(park[TARGET_COLS].describe().round(2))
print()

# Normalización (entrada y salida)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ─────────────────────────────────────────────
# Construcción del modelo
# ─────────────────────────────────────────────
def build_model(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64,  activation='relu'),
        layers.Dense(32,  activation='relu'),
        layers.Dense(2)   # 2 salidas: motor_UPDRS y total_UPDRS (lineal)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ─────────────────────────────────────────────
# Cross-validación 5 folds
# ─────────────────────────────────────────────
N_SPLITS   = 5
N_EPOCHS   = 100
BATCH_SIZE = 64

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

mse_motor, mse_total = [], []
r2_motor,  r2_total  = [], []
history_list = []

all_yte_motor, all_ypred_motor = [], []
all_yte_total, all_ypred_total = [], []

print(f"Entrenando MLP Regresión (Keras) con {N_SPLITS}-fold CV...")
print(f"Arquitectura: Dense(128) → Dense(64) → Dense(32) → Dense(2, lineal)")
print(f"Optimizer: Adam | Loss: MSE | Épocas: {N_EPOCHS}\n")

for fold, (tr, te) in enumerate(kf.split(X_scaled), 1):
    X_tr, y_tr = X_scaled[tr], y_scaled[tr]
    X_te, y_te = X_scaled[te], y_scaled[te]

    model = build_model(X.shape[1])
    hist = model.fit(
        X_tr, y_tr,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_te, y_te),
        verbose=0
    )
    history_list.append(hist)

    y_pred_scaled = model.predict(X_te, verbose=0)

    # Invertir normalización para métricas en escala original
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
    y_te_orig   = scaler_y.inverse_transform(y_te)

    m_mse = mean_squared_error(y_te_orig[:, 0], y_pred_orig[:, 0])
    t_mse = mean_squared_error(y_te_orig[:, 1], y_pred_orig[:, 1])
    m_r2  = r2_score(y_te_orig[:, 0], y_pred_orig[:, 0])
    t_r2  = r2_score(y_te_orig[:, 1], y_pred_orig[:, 1])

    mse_motor.append(m_mse); mse_total.append(t_mse)
    r2_motor.append(m_r2);   r2_total.append(t_r2)

    all_yte_motor.extend(y_te_orig[:, 0].tolist())
    all_ypred_motor.extend(y_pred_orig[:, 0].tolist())
    all_yte_total.extend(y_te_orig[:, 1].tolist())
    all_ypred_total.extend(y_pred_orig[:, 1].tolist())

    print(f"  Fold {fold}:  motor MSE={m_mse:.3f}  R²={m_r2:.3f} | "
          f"total MSE={t_mse:.3f}  R²={t_r2:.3f}")

print("\n=== Resultados Finales (escala original) ===")
print(f"  motor_UPDRS — MSE: {np.mean(mse_motor):.4f} ± {np.std(mse_motor):.4f}"
      f"  |  RMSE: {np.sqrt(np.mean(mse_motor)):.4f}"
      f"  |  R²: {np.mean(r2_motor):.4f}")
print(f"  total_UPDRS — MSE: {np.mean(mse_total):.4f} ± {np.std(mse_total):.4f}"
      f"  |  RMSE: {np.sqrt(np.mean(mse_total)):.4f}"
      f"  |  R²: {np.mean(r2_total):.4f}")

# ─────────────────────────────────────────────
# Gráficas
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

x_folds = np.arange(1, N_SPLITS + 1)

# Fila 1: motor_UPDRS
# 1) MSE por fold
axes[0, 0].bar(x_folds, mse_motor, color='mediumseagreen', edgecolor='black', alpha=0.85)
axes[0, 0].axhline(np.mean(mse_motor), color='red', linestyle='--',
                   label=f'Media={np.mean(mse_motor):.2f}')
axes[0, 0].set_title('motor_UPDRS — MSE por Fold', fontsize=11)
axes[0, 0].set_xlabel('Fold'); axes[0, 0].set_ylabel('MSE')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# 2) Curva de pérdida
for i, hist in enumerate(history_list):
    axes[0, 1].plot(hist.history['val_loss'], alpha=0.7, label=f'Fold {i+1}')
axes[0, 1].set_title('Pérdida de Validación (MSE normalizado)', fontsize=11)
axes[0, 1].set_xlabel('Época'); axes[0, 1].set_ylabel('MSE (normalizado)')
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

# 3) Real vs. Predicho — motor_UPDRS
axes[0, 2].scatter(all_yte_motor, all_ypred_motor, alpha=0.2, s=5, color='mediumseagreen')
lim = [min(all_yte_motor), max(all_yte_motor)]
axes[0, 2].plot(lim, lim, 'r--', linewidth=1.5, label='Ideal')
axes[0, 2].set_title(f'Real vs. Predicho — motor_UPDRS\nR²={np.mean(r2_motor):.3f}', fontsize=11)
axes[0, 2].set_xlabel('Real'); axes[0, 2].set_ylabel('Predicho')
axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

# Fila 2: total_UPDRS
axes[1, 0].bar(x_folds, mse_total, color='mediumpurple', edgecolor='black', alpha=0.85)
axes[1, 0].axhline(np.mean(mse_total), color='red', linestyle='--',
                   label=f'Media={np.mean(mse_total):.2f}')
axes[1, 0].set_title('total_UPDRS — MSE por Fold', fontsize=11)
axes[1, 0].set_xlabel('Fold'); axes[1, 0].set_ylabel('MSE')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

# R² por fold (motor vs total)
axes[1, 1].plot(x_folds, r2_motor, 'o-', label='motor_UPDRS', color='mediumseagreen', linewidth=2)
axes[1, 1].plot(x_folds, r2_total, 's-', label='total_UPDRS', color='mediumpurple', linewidth=2)
axes[1, 1].set_title('R² por Fold', fontsize=11)
axes[1, 1].set_xlabel('Fold'); axes[1, 1].set_ylabel('R²')
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].scatter(all_yte_total, all_ypred_total, alpha=0.2, s=5, color='mediumpurple')
lim2 = [min(all_yte_total), max(all_yte_total)]
axes[1, 2].plot(lim2, lim2, 'r--', linewidth=1.5, label='Ideal')
axes[1, 2].set_title(f'Real vs. Predicho — total_UPDRS\nR²={np.mean(r2_total):.3f}', fontsize=11)
axes[1, 2].set_xlabel('Real'); axes[1, 2].set_ylabel('Predicho')
axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

plt.suptitle("MLP Regresión — Parkinson's Telemonitoring (5-fold CV)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('ejercicio2_3.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfica guardada: ejercicio2_3.png")