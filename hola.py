import numpy as np
import pandas as pd
import os
import warnings
import pickle  # Para exportar el modelo
from joblib import dump, load  # Alternativa para exportar

warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Cargar datasets
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        if filename == 'df_train.csv':
            df_train = pd.read_csv(os.path.join(dirname, filename))
        if filename == 'df_test.csv':
            df_test = pd.read_csv(os.path.join(dirname, filename))

if not df_train.empty:
    print('Train dataframe imported successfully!')
if not df_test.empty:
    print('Test dataframe imported successfully!')

df_train.head()

# Análisis inicial
print('Analysing the shape of our splitted datasets')
print('Train: {} lines | Test: {} lines'.format(df_train.shape[0], df_test.shape[0]))

print('Analysing columns types:\n')
df_train.info()

# Preprocesamiento de datos
X_train_raw = df_train.drop(columns=['date', 'price', 'month', 'bedrooms'])
X_test_raw = df_test.drop(columns=['date', 'price', 'month', 'bedrooms'])
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
X_train.set_index(X_train_raw.index, inplace=True)

X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)
X_test.set_index(X_test_raw.index, inplace=True)

X_train.head()

# Entrenamiento del modelo
model_1 = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(model_1.summary())

# Análisis de multicolinealidad
vif = pd.DataFrame({'feature':X_train.columns})
vif['VIF'] = [variance_inflation_factor((X_train * 1).values, i) for i in range(X_train.shape[1])]
vif.sort_values(by= 'VIF', ascending=False)

# Residuales
sns.residplot(x=model_1.fittedvalues, y=model_1.resid, lowess=True, color='red', line_kws={'color': 'blue'})
plt.title('Homoscedasticity')
plt.show()

statistic, p_value, _, __ = het_breuschpagan(model_1.resid, model_1.model.exog)
print(p_value)

model_robust = model_1.get_robustcov_results()
print(model_robust.summary())
model_robust.resid.mean() #very low

sns.distplot(model_robust.resid, kde=True)
plt.title('Normality of residuals')
plt.show()

sm.ProbPlot(model_robust.resid).qqplot(line='s');

durbin_watson(model_robust.resid)

# Predicciones y métricas
pred = model_robust.predict(sm.add_constant(X_test))

print(f'R² score: {r2_score(y_test, pred)*100}')
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, pred))
print("Mean Absolute Percentage Error:", np.mean(np.abs((y_test-pred) / y_test)) * 100)

# Exportar el modelo
export_path = 'modelo_robusto.joblib'  # Cambiar a .pkl si usas pickle
dump(model_robust, export_path)
print(f"Modelo exportado con éxito a {export_path}")

# Cómo cargar el modelo en otro momento
# model_loaded = load(export_path)
# print("Modelo cargado con éxito.")
