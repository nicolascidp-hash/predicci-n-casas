import pickle
import json
import numpy as np
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def cargar_pipeline():
    ruta = os.path.join(BASE_DIR, 'models', 'pipeline_gb.pkl')
    with open(ruta, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def cargar_feature_names():
    ruta = os.path.join(BASE_DIR, 'models', 'feature_names.json')
    with open(ruta, 'r') as f:
        return json.load(f)

def sanitize_name(name):
    s = re.sub(r'\s+', '_', name)
    s = re.sub(r'[^0-9a-zA-Z_]', '_', s)
    return s

def sanitize_features_for_form(feature_names):
    out = []
    for orig in feature_names:
        field = sanitize_name(orig)
        label = orig.replace('_', ' ').capitalize()
        out.append({'orig': orig, 'field': field, 'label': label})
    return out

def predecir(pipeline, valores_en_orden, feature_names):
    # valores_en_orden ya está en el mismo orden que feature_names
    X_arr = np.array(valores_en_orden).reshape(1, -1)
    pred = pipeline.predict(X_arr)[0]
    return float(pred)