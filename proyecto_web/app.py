from flask import Flask, render_template, request
from utils.model_loader import cargar_pipeline, cargar_feature_names, predecir
import os

app = Flask(__name__)

# Cargar modelo y nombres de columnas al iniciar
# Asegúrate de que las carpetas 'models' y 'utils' estén donde corresponde
pipeline = cargar_pipeline()
feature_names = cargar_feature_names()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    resultado = None
    error = None

    if request.method == 'POST':
        try:
            # 1. Obtener datos numéricos básicos del formulario
            # Usamos float() para convertir el texto a número
            input_data = {
                'yearly_median_income': float(request.form.get('yearly_median_income')),
                'housing_median_age': float(request.form.get('housing_median_age')),
                'median_rooms': float(request.form.get('median_rooms')),
                'median_bedrooms': float(request.form.get('median_bedrooms')),
                'households': float(request.form.get('households'))
            }

            # 2. Manejar el One-Hot Encoding manualmente (Ubicación)
            prox = request.form.get('ocean_proximity')
            
            # Inicializamos todas las columnas de ubicación en 0
            # IMPORTANTE: Estos nombres deben coincidir EXACTAMENTE con tu feature_names.json
            input_data['ocean_proximity_INLAND'] = 0
            input_data['ocean_proximity_ISLAND'] = 0
            input_data['ocean_proximity_NEAR_BAY'] = 0
            input_data['ocean_proximity_NEAR_OCEAN'] = 0
            
            # Activamos solo la columna que seleccionó el usuario
            if prox == 'INLAND': 
                input_data['ocean_proximity_INLAND'] = 1
            elif prox == 'ISLAND': 
                input_data['ocean_proximity_ISLAND'] = 1
            elif prox == 'NEAR_BAY': 
                input_data['ocean_proximity_NEAR_BAY'] = 1
            elif prox == 'NEAR_OCEAN': 
                input_data['ocean_proximity_NEAR_OCEAN'] = 1
            # Nota: Si seleccionan '<1H OCEAN', todas quedan en 0 (Categoría de referencia)

            # 3. Ordenar la lista de valores exactamente como lo espera el modelo
            # El modelo es "tonto", no sabe de nombres, solo espera una lista de números en orden exacto
            valores = []
            for feature in feature_names:
                # Si alguna columna del JSON no está en nuestro input_data, esto daría error.
                # Como lo hemos programado manual, nos aseguramos de cubrir todas.
                valores.append(input_data[feature])

            # 4. Predecir
            resultado = predecir(pipeline, valores, feature_names)

        except Exception as e:
            error = f"Ocurrió un error al procesar los datos: {str(e)}"
            print(f"DEBUG - Error detallado: {e}") 

    # Renderizamos la plantilla pasando el resultado (o el error si hubo)
    return render_template('prediccion.html', resultado=resultado, error=error)

if __name__ == "__main__":
    # Configuración para ejecutar localmente
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)