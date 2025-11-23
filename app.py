import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# Librerías de Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
# Usamos el archivo que acabas de subir
DATA_FILE = 'dataset_sinescalar.csv' 

model_pipeline = None

# Definimos las columnas numéricas que necesitan escalado
NUMERIC_FEATURES = [
    'housing_median_age', 
    'households', 
    'yearly_median_income', 
    'median_rooms', 
    'median_bedrooms'
]

# Definimos las columnas de Categoría que ya vienen en tu CSV
CAT_FEATURES = [
    'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND',
    'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN'
    # Nota: Tu CSV no parece tener la columna '<1H OCEAN', así que la trataremos como la base (todo 0)
]

def train_model():
    global model_pipeline
    
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: No encuentro el archivo {DATA_FILE}")
        return

    print(f"Cargando {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Definir Features (X) y Objetivo (y)
    # Combinamos las listas de columnas que definimos arriba
    all_features = NUMERIC_FEATURES + CAT_FEATURES
    
    # Verificación de seguridad: si falta alguna columna, la creamos en 0
    for col in all_features:
        if col not in df.columns:
            df[col] = 0

    X = df[all_features]
    y = df['median_house_value']

    print("Entrenando Pipeline (Escalador Automático + Random Forest)...")
    
    # 1. El Preprocesador: Escala SOLO los números, deja pasar los 0s y 1s de las categorías
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES)
        ],
        remainder='passthrough' 
    )

    # 2. La Tubería (Pipeline): Preprocesa -> Predice
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X, y)
    print("✅ Modelo entrenado correctamente con datos reales.")

# Entrenar al arrancar
train_model()

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_pipeline:
        return jsonify({'error': 'Modelo no cargado'}), 500

    try:
        data = request.get_json()
        
        # Construir los datos de entrada
        input_data = {}
        
        # Datos numéricos
        input_data['housing_median_age'] = [float(data['housing_median_age'])]
        input_data['households'] = [float(data['households'])]
        input_data['yearly_median_income'] = [float(data['yearly_median_income'])]
        input_data['median_rooms'] = [float(data['median_rooms'])]
        input_data['median_bedrooms'] = [float(data['median_bedrooms'])]
        
        # Datos categóricos (Convertir selección a One-Hot para que coincida con el CSV)
        prox = data['ocean_proximity']
        
        # Poner todas las categorías en 0 primero
        for col in CAT_FEATURES:
            input_data[col] = [0]
            
        # Activar la seleccionada
        if prox == 'INLAND': input_data['ocean_proximity_INLAND'] = [1]
        elif prox == 'ISLAND': input_data['ocean_proximity_ISLAND'] = [1]
        elif prox == 'NEAR BAY': input_data['ocean_proximity_NEAR BAY'] = [1]
        elif prox == 'NEAR OCEAN': input_data['ocean_proximity_NEAR OCEAN'] = [1]
        
        # Convertir a DataFrame
        df_input = pd.DataFrame(input_data)
        
        # Ordenar columnas igual que en el entrenamiento
        df_input = df_input[NUMERIC_FEATURES + CAT_FEATURES]
        
        # Predecir (El pipeline escala automáticamente)
        prediction = model_pipeline.predict(df_input)[0]
        
        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)