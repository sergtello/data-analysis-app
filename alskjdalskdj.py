from joblib import load
import numpy as np

# Cargar el modelo
model_path = 'modelo_robusto.joblib'
model = load(model_path)

# Ejemplo de datos escalados (ajusta según tus columnas y preprocesamiento)
ejemplo_escala = np.array([[3, 7, 0, 120.0, 0, 0, 0, 2, 0, 1, 2]])

# Realizar predicción
pred_price = model.predict(ejemplo_escala)  # Sin agregar constante
print("Predicción manual:", pred_price[0])
