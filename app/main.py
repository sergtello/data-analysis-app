import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar el modelo y el escalador previamente exportados
model_path = 'models/model_linear.joblib'
scaler_path = 'models/scaler.joblib'

model = load(model_path)
scaler = load(scaler_path)

# Configurar las columnas que se usaron en el modelo
features = ['bedrooms', 'grade', 'has_basement', 'living_in_m2', 'renovated',
            'nice_view', 'perfect_condition', 'real_bathrooms',
            'has_lavatory', 'single_floor', 'month', 'quartile_zone']

# Título del dashboard
st.title("Análisis del Modelo de Predicción de Precios de Casas")

st.write("""
Este dashboard utiliza un dataset preprocesado para analizar el rendimiento del modelo de regresión lineal entrenado.
""")

# Cargar el dataset preprocesado
data_path = './datasets/df_train.csv'

try:
    # Cargar el dataset
    data = pd.read_csv(data_path)
    st.success("Dataset cargado exitosamente")

    # Mostrar vista general del dataset
    st.subheader("Vista General del Dataset")
    st.write(data.head())

    # Descripción estadística
    st.subheader("Estadísticas Descriptivas")
    st.write(data.describe())

    # Distribución de las características clave
    st.subheader("Distribución de las Características")
    selected_feature = st.selectbox("Selecciona una característica para graficar:", features)
    st.bar_chart(data[selected_feature].value_counts())

    # Evaluación del modelo
    st.subheader("Evaluación del Modelo")

    # Verificar si la variable objetivo está en el dataset
    if 'price' in data.columns:
        X = data[features]
        y = data['price']

        # Escalar las características
        X_scaled = scaler.transform(X)

        # Agregar constante
        X_scaled_with_const = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

        # Predicciones del modelo
        predictions = model.predict(X_scaled_with_const)

        # Calcular métricas
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Mostrar métricas
        st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Comparación de predicciones y valores reales
        st.subheader("Comparación: Predicciones vs Valores Reales")
        comparison_df = pd.DataFrame({'Real Price': y, 'Predicted Price': predictions})
        st.write(comparison_df.head(10))
        st.line_chart(comparison_df.head(50))

except FileNotFoundError:
    st.error(f"No se encontró el archivo {data_path}. Por favor, asegúrate de que está en la ruta especificada.")
