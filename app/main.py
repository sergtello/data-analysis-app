import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# Cargar el modelo y el escalador previamente exportados
model_path = 'models/modelo_robusto.joblib'
scaler_path = 'models/scaler.joblib'

model = load(model_path)
scaler = load(scaler_path)

# Configurar las columnas que se usaron en el modelo
features = ['bedrooms', 'grade', 'has_basement', 'living_in_m2', 'renovated',
            'nice_view', 'perfect_condition', 'real_bathrooms',
            'has_lavatory', 'single_floor', 'month', 'quartile_zone']

# Título del dashboard
st.title("Predicción del Precio de Casas")

st.write("""
Este dashboard predice el precio de una casa en función de características como habitaciones, baños, tamaño y ubicación.
Por favor, ingresa los datos solicitados en el menú lateral.
""")

# Sección de entrada de datos
st.sidebar.header("Ingrese las características de la casa")

# Crear campos para entrada de datos (deben coincidir con las características del modelo)
input_data = {
    'bedrooms': st.sidebar.number_input('Habitaciones', min_value=0, step=1, value=3),
    'grade': st.sidebar.slider('Calidad de construcción (1-13)', min_value=1, max_value=13, value=7),
    'has_basement': st.sidebar.selectbox('¿Tiene sótano?', options=[True, False], format_func=lambda x: "Sí" if x else "No"),
    'living_in_m2': st.sidebar.number_input('Área habitable (m²)', min_value=0.0, step=1.0, value=120.0),
    'renovated': st.sidebar.selectbox('¿Está renovada?', options=[True, False], format_func=lambda x: "Sí" if x else "No"),
    'nice_view': st.sidebar.selectbox('¿Tiene buena vista?', options=[True, False], format_func=lambda x: "Sí" if x else "No"),
    'perfect_condition': st.sidebar.selectbox('¿En condición perfecta?', options=[True, False], format_func=lambda x: "Sí" if x else "No"),
    'real_bathrooms': st.sidebar.number_input('Baños reales', min_value=0, step=1, value=2),
    'has_lavatory': st.sidebar.selectbox('¿Tiene lavatorio?', options=[True, False], format_func=lambda x: "Sí" if x else "No"),
    'single_floor': st.sidebar.selectbox('¿Es de un solo piso?', options=[True, False], format_func=lambda x: "Sí" if x else "No"),
    'quartile_zone': st.sidebar.slider('Zona por cuartil (1-4)', min_value=1, max_value=4, value=2),
    'date': st.sidebar.date_input("Fecha de la venta", value=pd.Timestamp.now())
}

# Convertir la fecha a mes
input_data['month'] = pd.to_datetime(input_data['date']).month

# Eliminar la columna 'date' ya que no se utiliza en el modelo
del input_data['date']

# Convertir datos categóricos booleanos a enteros
for col in ['has_basement', 'renovated', 'nice_view', 'perfect_condition', 'has_lavatory', 'single_floor']:
    input_data[col] = int(input_data[col])

# Convertir la entrada a un DataFrame
input_df = pd.DataFrame([input_data])

# Escalar los datos
input_scaled = scaler.transform(input_df[features])

# Agregar la constante manualmente
input_scaled_with_const = np.c_[np.ones((input_scaled.shape[0], 1)), input_scaled]

# Verificar las dimensiones después de agregar la constante
st.write("Dimensiones después de agregar constante:", input_scaled_with_const.shape)

# Mostrar los datos ingresados
st.subheader("Datos ingresados:")
st.write(input_df)

# Mostrar los valores escalados
st.write("Valores escalados utilizados en la predicción (incluye constante):")
st.write(input_scaled_with_const)

# Realizar la predicción
if st.button("Predecir Precio"):
    try:
        # Predecir usando el modelo con la constante incluida
        pred_price = model.predict(input_scaled_with_const)
        st.subheader("Predicción del Precio")
        st.write(f"El precio estimado de la casa es: ${pred_price[0]:,.2f}")
    except ValueError as e:
        st.error(f"Error en la predicción: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")
