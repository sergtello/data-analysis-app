from joblib import load

# Cargar el modelo
model_path = 'modelo_robusto.joblib'
model = load(model_path)

# Revisar los coeficientes del modelo
try:
    print("Coeficientes del modelo:", model.params)  # Para statsmodels
    print("Intercepción:", model.params[0])  # Verificar si la constante es significativa
except AttributeError:
    print("El modelo no tiene coeficientes accesibles directamente. Verifica cómo fue entrenado.")
