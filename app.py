import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# --- CAMBIO: Importamos Gradient Boosting ---
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
DATA_FILE = 'dataset_sinescalar.csv' 

model_pipeline = None

# Columnas numéricas
NUMERIC_FEATURES = [
    'housing_median_age', 
    'households', 
    'yearly_median_income', 
    'median_rooms', 
    'median_bedrooms'
]

# Columnas categóricas (One-Hot)
CAT_FEATURES = [
    'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND',
    'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN'
]

def train_model():
    global model_pipeline
    
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: No encuentro el archivo {DATA_FILE}")
        return

    print(f"Cargando {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Rellenar columnas faltantes por seguridad
    all_features = NUMERIC_FEATURES + CAT_FEATURES
    for col in all_features:
        if col not in df.columns:
            df[col] = 0

    X = df[all_features]
    y = df['median_house_value']

    print("Entrenando Pipeline (Escalador + Gradient Boosting)...")
    
    # 1. Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES)
        ],
        remainder='passthrough' 
    )

    # 2. Pipeline con Gradient Boosting
    # Usamos parámetros estándar robustos (n_estimators=100, learning_rate=0.1)
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
    ])

    model_pipeline.fit(X, y)
    print("✅ Modelo Gradient Boosting entrenado correctamente.")

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
        
        # Construir entrada
        input_data = {}
        
        input_data['housing_median_age'] = [float(data['housing_median_age'])]
        input_data['households'] = [float(data['households'])]
        input_data['yearly_median_income'] = [float(data['yearly_median_income'])]
        input_data['median_rooms'] = [float(data['median_rooms'])]
        input_data['median_bedrooms'] = [float(data['median_bedrooms'])]
        
        prox = data['ocean_proximity']
        for col in CAT_FEATURES:
            input_data[col] = [0]
            
        if prox == 'INLAND': input_data['ocean_proximity_INLAND'] = [1]
        elif prox == 'ISLAND': input_data['ocean_proximity_ISLAND'] = [1]
        elif prox == 'NEAR BAY': input_data['ocean_proximity_NEAR BAY'] = [1]
        elif prox == 'NEAR OCEAN': input_data['ocean_proximity_NEAR OCEAN'] = [1]
        
        df_input = pd.DataFrame(input_data)
        df_input = df_input[NUMERIC_FEATURES + CAT_FEATURES]
        
        # Predecir
        prediction = model_pipeline.predict(df_input)[0]
        
        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
