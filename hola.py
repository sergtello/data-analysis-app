import numpy as np
import pandas as pd
import os
import warnings
from joblib import dump
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.simplefilter("ignore")

# Cargar datasets
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        if filename == 'df_train.csv':
            df_train = pd.read_csv(os.path.join(dirname, filename))
        if filename == 'df_test.csv':
            df_test = pd.read_csv(os.path.join(dirname, filename))

# Verificar que los datasets se cargaron correctamente
if not df_train.empty:
    print('Train dataframe imported successfully!')
if not df_test.empty:
    print('Test dataframe imported successfully!')

# Preprocesamiento de datos
X_train_raw = df_train.drop(columns=['date', 'price'])  # Excluir columnas irrelevantes
X_test_raw = df_test.drop(columns=['date', 'price'])  # Excluir columnas irrelevantes
y_train = df_train['price']
y_test = df_test['price']

# Ajustar el escalador con los datos de entrenamiento
scaler = StandardScaler()
scaler.fit(X_train_raw)

# Exportar el escalador
scaler_path = 'scaler.joblib'
dump(scaler, scaler_path)
print(f"Scaler exportado con éxito a {scaler_path}")

# Transformar los datos de entrenamiento y prueba
X_train = pd.DataFrame(scaler.transform(X_train_raw), columns=X_train_raw.columns)
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

# Entrenamiento del modelo
model_robust = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(model_robust.summary())

# Exportar el modelo
export_path = 'modelo_robusto.joblib'
dump(model_robust, export_path)
print(f"Modelo exportado con éxito a {export_path}")
