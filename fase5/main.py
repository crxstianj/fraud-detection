from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(title="Elliptic Envelope API with LDA")

# Habilitar CORS (para frontend externo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta de los modelos
base_dir = os.path.dirname(__file__)
elliptic_model_path = os.path.join(base_dir, "..", "elliptic_envelope_model.pkl")
lda_model_path = os.path.join(base_dir, "..", "lda_model.pkl")

# Cargar modelos
try:
    elliptic_model = joblib.load(elliptic_model_path)
    lda_model = joblib.load(lda_model_path)
except Exception as e:
    raise RuntimeError(f"Error al cargar modelos: {e}")

# Esquema de entrada
class InputData(BaseModel):
    data: list[list[float]]  # matriz n_samples x n_features

@app.post("/predict")
def predict(input_data: InputData):
    try:
        X = np.array(input_data.data)

        # Transformación LDA
        X_transformed = lda_model.transform(X)

        # Predicción con EllipticEnvelope
        prediction = elliptic_model.predict(X_transformed).tolist()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
