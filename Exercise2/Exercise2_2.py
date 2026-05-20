"""
Ejercicio 2.2 — MLP para clasificación de 4 clases (Keras)
Dataset: misterious_data_4.txt (4 clases)
Evaluación: Cross-validación estratificada de 5 folds
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
# Carga y preparación de datos
# ─────────────────────────────────────────────
data = np.loadtxt('Exercise2/misterious_data_4.txt', delimiter='\t')
X = data[:, 1:].astype(np.float32)
y = data[:, 0].astype(int) - 1   # clases 1-4 → 0-3

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_classes = len(np.unique(y))
print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características, {n_classes} clases")
for c in range(n_classes):
    print(f"  Clase {c+1}: {(y == c).sum()} muestras")
print()

# ─────────────────────────────────────────────
# Construcción del modelo
# ─────────────────────────────────────────────
def build_model(input_dim, n_classes):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ─────────────────────────────────────────────
# Cross-validación estratificada 5 folds
# ─────────────────────────────────────────────
N_SPLITS = 5
N_EPOCHS = 100
BATCH_SIZE = 32

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_acc = []
all_y_true, all_y_pred = [], []
history_list = []

print(f"Entrenando MLP (Keras) con {N_SPLITS}-fold CV estratificada...")
print(f"Arquitectura: Dense(128,ReLU) → Dropout(0.3) → Dense(64,ReLU) → Dense({n_classes},Softmax)")
print(f"Optimizer: Adam | Loss: Sparse Categorical CE | Épocas: {N_EPOCHS}\n")

for fold, (tr, te) in enumerate(kf.split(X_scaled, y), 1):
    X_tr, y_tr = X_scaled[tr], y[tr]
    X_te, y_te = X_scaled[te], y[te]

    model = build_model(X.shape[1], n_classes)
    hist = model.fit(
        X_tr, y_tr,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_te, y_te),
        verbose=0
    )
    history_list.append(hist)

    _, acc = model.evaluate(X_te, y_te, verbose=0)
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)

    fold_acc.append(acc)
    all_y_true.extend(y_te.tolist())
    all_y_pred.extend(y_pred.tolist())
    print(f"  Fold {fold}: Accuracy = {acc:.4f}")

print(f"\nAccuracy media CV: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
print("\nReporte de Clasificación (todos los folds):")
print(classification_report(all_y_true, all_y_pred,
                             target_names=[f'Clase {i+1}' for i in range(n_classes)]))

# ─────────────────────────────────────────────
# Gráficas
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 1) Accuracy por fold
axes[0].bar(range(1, N_SPLITS + 1), fold_acc,
            color='darkorange', alpha=0.85, edgecolor='black')
axes[0].axhline(np.mean(fold_acc), color='red', linestyle='--',
                label=f'Media = {np.mean(fold_acc):.3f}')
axes[0].set_title('Accuracy por Fold (5-fold CV)', fontsize=12)
axes[0].set_xlabel('Fold'); axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1.05); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 2) Curvas de accuracy de validación
for i, hist in enumerate(history_list):
    axes[1].plot(hist.history['val_accuracy'], alpha=0.7, label=f'Fold {i+1}')
axes[1].set_title('Accuracy de Validación por Fold', fontsize=12)
axes[1].set_xlabel('Época'); axes[1].set_ylabel('Accuracy')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# 3) Matriz de confusión
cm = confusion_matrix(all_y_true, all_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
            xticklabels=[f'C{i+1}' for i in range(n_classes)],
            yticklabels=[f'C{i+1}' for i in range(n_classes)])
axes[2].set_title('Matriz de Confusión (todos los folds)', fontsize=12)
axes[2].set_xlabel('Predicho'); axes[2].set_ylabel('Real')

plt.tight_layout()
plt.savefig('ejercicio2_2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfica guardada: ejercicio2_2.png")