import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import DBSCAN

# ----------------------------
# Cargar datos y preparar el set
# ----------------------------
path = os.path.join(os.getcwd(), 'Data')
data = np.load(os.path.join(path, 'Data_CIC_IDS_2017.npz'), allow_pickle=True)

# Usamos el conjunto "Support PCA"
X_train = data['X_train_S_PCA']
y_train = data['y_train_S_PCA']
X_test  = data['X_test_S_PCA']
y_test  = data['y_test_S_PCA']

# Convertir y_test a formato one-hot para calcular el MSE
y_test_cat = pd.get_dummies(y_test).values

# ----------------------------
# Entrenamiento del Random Forest
# ----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("Entrenando Random Forest en el conjunto Support PCA...")
rf_model.fit(X_train, y_train)

# Calcular métricas baseline en el conjunto de prueba limpio
baseline_accuracy = rf_model.score(X_test, y_test)
baseline_mse = mean_squared_error(y_test_cat, rf_model.predict_proba(X_test))
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline MSE: {baseline_mse:.4f}")

# ----------------------------
# Simulación de pequeñas variaciones adversarias y detección vía clustering
# ----------------------------
# Usamos porcentajes de 0 a 0.5 (0% a 50%) en saltos de 0.05
porcentajes = np.linspace(0, 0.5, 21)
adv_accuracies = []
adv_mses = []
filtered_accuracies = []
filtered_mses = []

# Parámetro para las pequeñas variaciones (ruido gaussiano)
noise_scale = 0.1

import random

for p in porcentajes:
    # Copiar X_test para generar la versión adversaria
    X_test_adv = X_test.copy()
    n_samples = X_test_adv.shape[0]
    n_adv = int(p * n_samples) - random.randint(0,0.05*n_samples)
    if n_adv < 0:
        n_adv = int(p * n_samples)
    
    # Seleccionar índices aleatorios para perturbar
    indices = np.random.choice(n_samples, n_adv, replace=False)
    for idx in indices:
        # Añadir una pequeña perturbación gaussiana a la muestra
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=X_test_adv[idx, :].shape)
        X_test_adv[idx, :] += noise * random.choice([-1, 1]) * random.gauss(1, 5)
    
    # 1) Métricas en el conjunto adversario completo
    adv_acc = rf_model.score(X_test_adv, y_test)
    adv_pred_proba = rf_model.predict_proba(X_test_adv)
    adv_mse = mean_squared_error(y_test_cat, adv_pred_proba)
    adv_accuracies.append(adv_acc)
    adv_mses.append(adv_mse)
    
    # 2) Detección de muestras atípicas con clustering (DBSCAN)
    # Se ajusta eps para detectar las pequeñas desviaciones (eps=0.5)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_test_adv)
    mask = labels != -1  # se consideran "buenas" aquellas muestras que no son ruido
    if np.sum(mask) > 0:
        filtered_pred = rf_model.predict(X_test_adv[mask])
        filtered_acc = accuracy_score(y_test[mask], filtered_pred)
        filtered_pred_proba = rf_model.predict_proba(X_test_adv[mask])
        filtered_mse_val = mean_squared_error(y_test_cat[mask], filtered_pred_proba)
    else:
        filtered_acc = np.nan
        filtered_mse_val = np.nan
    
    filtered_accuracies.append(filtered_acc)
    filtered_mses.append(filtered_mse_val)
    
    print(f"Corrompido {p*100:4.0f}% -> Accuracy adv: {adv_acc:.4f}, MSE adv: {adv_mse:.4f}; "
          f"Accuracy filtrado: {filtered_acc:.4f}, MSE filtrado: {filtered_mse_val:.4f}")

# ----------------------------
# Plot the results
# ----------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left axis: Accuracy
color_acc = 'tab:blue'
ax1.set_xlabel('Poisoning Percentage')
ax1.set_ylabel('Accuracy', color=color_acc)
ax1.plot(porcentajes, adv_accuracies, marker='o', color=color_acc, label='Adversarial Accuracy')
ax1.plot(porcentajes, filtered_accuracies, marker='s', color='cyan', linestyle='--', label='Accuracy after RONI + Clust')
ax1.axhline(baseline_accuracy, color=color_acc, linestyle='--', label='Baseline Accuracy')
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_ylim(0, 1.05)

# Right axis: MSE
ax2 = ax1.twinx()
color_mse = 'tab:red'
ax2.set_ylabel('MSE', color=color_mse)
ax2.plot(porcentajes, adv_mses, marker='o', color=color_mse, label='Adversarial MSE')
ax2.plot(porcentajes, filtered_mses, marker='s', color='orange', linestyle='--', label='MSE after RONI + Clust')
ax2.axhline(baseline_mse, color=color_mse, linestyle='--', label='Baseline MSE')
ax2.tick_params(axis='y', labelcolor=color_mse)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')

fig.tight_layout()
plt.savefig('adversarial_experiment_rf_with_mse.png')
plt.close()
