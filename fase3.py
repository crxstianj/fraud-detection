import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib

# 1. Cargar datos
df = pd.read_csv("data/creditcard.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values  # 1 = fraude, 0 = normal

# 2. Reducción de dimensionalidad con LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

# 3. Calcular fracción de fraudes
outlier_fraction = np.mean(y == 1)
print(f"Fracción de fraudes: {outlier_fraction:.4f}")

# 4. Entrenar modelos
iso = IsolationForest(contamination=outlier_fraction, random_state=42)
iso.fit(X_lda)

elliptic = EllipticEnvelope(contamination=outlier_fraction, random_state=42)
elliptic.fit(X_lda)

# NOTA: LOF no tiene metodo fit separado, se entrena al predecir

# 5. Guardar modelos y datos procesados
joblib.dump(X_lda, "models/X_lda.joblib")
joblib.dump(y, "models/y_true.joblib")
joblib.dump(lda, "models/lda_model.joblib")
joblib.dump(iso, "models/iso_model.joblib")
joblib.dump(elliptic, "models/y_pred.joblib")

print("Modelos y datos guardados.")
