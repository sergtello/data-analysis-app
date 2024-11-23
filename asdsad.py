from joblib import load

model_path = 'modelo_robusto.joblib'
scaler_path = 'scaler.joblib'

model = load(model_path)
scaler = load(scaler_path)
print(model.params)