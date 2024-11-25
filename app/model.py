import pandas as pd
import os
import warnings
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.simplefilter("ignore")

df_train = None
df_test = None

# Cargar datasets
for dirname, _, filenames in os.walk('./datasets/'):
    for filename in filenames:
        if filename == 'df_train.csv':
            df_train = pd.read_csv(os.path.join(dirname, filename))
        if filename == 'df_test.csv':
            df_test = pd.read_csv(os.path.join(dirname, filename))

# Verificar que los datasets se cargaron correctamente
if df_train is not None and not df_train.empty:
    print('Train dataframe imported successfully!')
if df_test is not None and not df_test.empty:
    print('Test dataframe imported successfully!')

df_train_model = df_train.copy(deep=True)
df_test_model = df_test.copy(deep=True)

# Preprocesamiento de datos
X_train_raw = df_train_model.drop(columns=['date', 'price'])  # Excluir columnas irrelevantes
X_test_raw = df_test_model.drop(columns=['date', 'price'])  # Excluir columnas irrelevantes
y_train = df_train_model['price']
y_test = df_test_model['price']

# Ajustar el escalador con los datos de entrenamiento
scaler = StandardScaler()
scaler.fit(X_train_raw)

# Exportar el escalador
scaler_path = 'models/scaler.joblib'
dump(scaler, scaler_path)
print(f"Scaler exportado con éxito a {scaler_path}")

# Transformar los datos de entrenamiento y prueba
X_train = pd.DataFrame(scaler.transform(X_train_raw), columns=X_train_raw.columns)
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

# Entrenamiento del modelo Linear Regression
model_linear = sm.OLS(y_train, sm.add_constant(X_train)).fit()
model_linear.summary()

# Evaluar el modelo
y_pred_linear = model_linear.predict(sm.add_constant(X_test))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred_linear))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred_linear))
print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(y_test, y_pred_linear, squared=False))
print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(y_test, y_pred_linear))
print('Explained Variance Score:', metrics.explained_variance_score(y_test, y_pred_linear))
print('Max Error:', metrics.max_error(y_test, y_pred_linear))
print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_test, y_pred_linear))
print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_pred_linear))
print('R^2:', metrics.r2_score(y_test, y_pred_linear))
print('Mean Poisson Deviance:', metrics.mean_poisson_deviance(y_test, y_pred_linear))
print('Mean Gamma Deviance:', metrics.mean_gamma_deviance(y_test, y_pred_linear))

# Exportar el modelo
export_path_linear = 'models/model_linear.joblib'
dump(model_linear, export_path_linear)


# Entrenamiento del modelo Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Evaluar el modelo
y_pred_rf = model_rf.predict(X_test)

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred_rf))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred_rf))
print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(y_test, y_pred_rf, squared=False))
print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(y_test, y_pred_rf))
print('Explained Variance Score:', metrics.explained_variance_score(y_test, y_pred_rf))
print('Max Error:', metrics.max_error(y_test, y_pred_rf))
print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_test, y_pred_rf))
print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_pred_rf))
print('R^2:', metrics.r2_score(y_test, y_pred_rf))
print('Mean Poisson Deviance:', metrics.mean_poisson_deviance(y_test, y_pred_rf))
print('Mean Gamma Deviance:', metrics.mean_gamma_deviance(y_test, y_pred_rf))

# Exportar el modelo
export_path = 'models/model_rf.joblib'
dump(model_rf, export_path)
print(f"Modelo Random Forest exportado con éxito a {export_path}")
