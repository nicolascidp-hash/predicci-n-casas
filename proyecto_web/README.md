Proyecto de Predicción de Precios de Vivienda 

Descripción

Aplicación web desarrollada con Flask y Machine Learning para estimar el valor medio de viviendas en California basándose en características censales.

Objetivo

Proporcionar una herramienta intuitiva para predecir la variable median_house_value utilizando un modelo de Gradient Boosting entrenado.

Modelo Utilizado

Algoritmo: Gradient Boosting Regressor.

Entrenamiento: Se utilizó un pipeline con StandardScaler para normalizar datos numéricos.

Validación: El modelo fue validado con métricas R2 y RMSE.

Instrucciones de Navegación

Inicio: Pantalla de bienvenida.

Predicción: Formulario donde se ingresan los datos (Ingresos, Edad, Ubicación, etc.).

Resultado: Al enviar el formulario, se muestra el precio estimado en dólares.

Estructura del Proyecto

app.py: Servidor principal.

models/: Contiene el modelo serializado (.pkl) y metadatos.

static/: Estilos CSS.

templates/: Archivos HTML.

utils/: Lógica de carga del modelo.