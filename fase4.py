import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar datos y modelos
X_lda = joblib.load("models/X_lda.joblib")
y = joblib.load("models/y_true.joblib")
iso = joblib.load("models/iso_model.joblib")
elliptic = joblib.load("models/elliptic_model.joblib")

# 2. Obtener predicciones
# Isolation Forest
y_pred_iso = iso.predict(X_lda)
y_pred_iso = np.where(y_pred_iso == 1, 0, 1)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=np.mean(y == 1))
y_pred_lof = lof.fit_predict(X_lda)
y_pred_lof = np.where(y_pred_lof == 1, 0, 1)

# Elliptic Envelope
y_pred_ell = elliptic.predict(X_lda)
y_pred_ell = np.where(y_pred_ell == 1, 0, 1)

# 3. Evaluar y graficar
model_names = ["Isolation Forest", "Local Outlier Factor", "Elliptic Envelope"]
predictions = [y_pred_iso, y_pred_lof, y_pred_ell]

for name, y_pred in zip(model_names, predictions):
    print(f"\n--- {name} ---")
    print("Classification Report:")
    print(classification_report(y, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)

    # Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Anómalo"],
                yticklabels=["Normal", "Anómalo"])
    plt.title(f"Matriz de Confusión - {name}")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.tight_layout()
    plt.show()
