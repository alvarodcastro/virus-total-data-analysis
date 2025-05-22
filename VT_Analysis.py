import pandas as pd
import json
from datetime import datetime
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import folium
import pycountry
import numpy as np

# Load environment variables from a .env file
load_dotenv()

# Retrieve MongoDB connection string from environment variables
MONGO_URI = os.getenv('MONGO_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

if not MONGO_URI or not DATABASE_NAME or not COLLECTION_NAME:
    raise ValueError("Missing MongoDB connection string or database/collection name in environment variables.")

directory = "./VTAndroid"
data = []

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]  # New collection for dataframe

# Cargar los datos de la colección en un DataFrame de pandas
data = list(collection.find())
df = pd.DataFrame(data)


# ======= ANÁLISIS 1: Detección media por país y número de muestras =======

# Pipeline de agregación:
# $match: Filtra los documentos para incluir solo aquellos donde el campo submission.submitter_country existe.
# $group: Agrupa los documentos por el campo submission.submitter_country (el país del remitente de la presentación). 
# Para cada grupo, cuenta cuántos documentos hay ("count": {"$sum": 1}) y calcula la detección media ("deteccion_media": {"$avg": {"$divide": ["$positives", "$total"]}}).
# $sort: Ordena los resultados por el campo count en orden descendente (-1).

print("\nANALISIS 1: Detección media por país y número de muestras")
# 1. Agregación por país
agg_pais_detection = [
    {"$match": {"submission.submitter_country": {"$exists": True}}},
    {"$group": {
        "_id": "$submission.submitter_country",
        "num_muestras": {"$sum": 1},
        "deteccion_media": {"$avg": {"$divide": ["$positives", "$total"]}}
    }},
    {"$sort": {"num_muestras": -1}}
]

# Resultado: Una lista de países (no nulos) y la cantidad de documentos asociados a cada uno, ordenados del país con más documentos al que tiene menos.
results = list(collection.aggregate(agg_pais_detection))
# Only for debugging
print("Datos agregados por país")
for result in results:
    print(f"País: {result['_id']}, Cantidad: {result['num_muestras']}, Detección media: {result['deteccion_media']}")

df = pd.DataFrame(results)
df.columns = ['country_code', 'count', 'avg_detection']

# 2. Convertir código ISO2 a nombre completo
def code_to_country_name(code):
    try:
        return pycountry.countries.get(alpha_2=code.upper()).name
    except:
        return None

df['country'] = df['country_code'].apply(code_to_country_name)
df = df.dropna(subset=['country'])

# 3. Crear mapa con folium (choropleth)
world_map = folium.Map(location=[20, 0], zoom_start=2)

choropleth = folium.Choropleth(
    geo_data='https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json',
    name='Muestras por país',
    data=df,
    columns=['country', 'count'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Cantidad de muestras enviadas',
).add_to(world_map)

folium.LayerControl().add_to(world_map)
world_map.save("mapa_muestras_por_pais.html")
print("Mapa guardado en: mapa_muestras_por_pais.html")

# ======= ANÁLISIS 2: Detección media por antivirus =======
print("\nANALISIS 2: Detección media por antivirus")
# 1. $project: crea un nuevo documento para cada entrada, seleccionando y calculando campos específicos:
# "sha256": 1 Incluye el campo sha256 en la salida.
# "positives": 1 Incluye el campo positives (probablemente el número de detecciones positivas).
# "total": 1 Incluye el campo total (posiblemente el total de motores de análisis).
# "ratio": {"$divide": ["$positives", "$total"]}
# # Crea un nuevo campo llamado ratio que es el resultado de dividir positives entre total (porcentaje de positivos).
# 2. $sort
# {"positives": -1} Ordena los documentos resultantes de mayor a menor según el campo positives. Así, los archivos con más positivos aparecen primero.
# 3. $limit
# {"$limit": 10} # Limita la salida a solo los 10 primeros documentos, es decir, los 10 archivos con más positivos.

agg_top_muestras = [
    {"$project": {
        "sha256": 1,
        "positives": 1,
        "total": 1,
        "ratio": {"$divide": ["$positives", "$total"]}
    }},
    {"$sort": {"positives": -1}},
    {"$limit": 10}
]

# Resultado:
# Selecciona los 10 archivos con más positivos, mostrando su hash (sha256), el número de positivos, el total de motores y la proporción de positivos respecto al total.
for result in collection.aggregate(agg_top_muestras):
    print(f"SHA256: {result['sha256']}, Positives: {result['positives']}, Total: {result['total']}, Ratio: {result['ratio']}")

# ======= ANÁLISIS 3: Detección media por motor de análisis =======
print("\nANALISIS 3: Detección media por motor de análisis")
# 1. $project: crea un nuevo documento para cada entrada, seleccionando y calculando campos específicos:
# "scans": 1 Incluye el campo scans en la salida.
# 2. $project: convierte el campo scans (que es un objeto) en un array de pares clave-valor, donde cada par representa un motor de análisis y su resultado.
# "scans_array": {"$objectToArray": "$scans"} Convierte el objeto scans en un array de pares clave-valor.
# 3. $unwind: descompone el array scans_array en documentos individuales, uno por cada motor de análisis.
# 4. $group: agrupa los documentos por el campo scans_array.k (el nombre del motor de análisis).
# Para cada grupo, cuenta cuántas detecciones positivas hay ("detecciones": {"$sum": {"$cond": [{"$eq": ["$scans_array.v.detected", True]}, 1, 0]}}) y cuenta el total de escaneos realizados por ese motor ("total_escaneos": {"$sum": 1}).
# 5. $project: selecciona los campos detecciones y total_escaneos, y calcula el ratio de detecciones respecto al total de escaneos.
# "ratio": {"$divide": ["$detecciones", "$total_escaneos"]} Calcula el ratio de detecciones respecto al total de escaneos.
# 6. $sort: ordena los resultados por el campo detecciones en orden descendente (-1).
# {"detecciones": -1} Ordena los documentos resultantes de mayor a menor según el campo detecciones. Así, los motores con más detecciones aparecen primero.
# 7. $limit: {"$limit": 10} Limita la salida a solo los 10 primeros documentos, es decir, los 10 motores de análisis con más detecciones.
agg_por_motor = [
    {"$project": {"scans": 1}},
    {"$project": {
        "scans_array": {"$objectToArray": "$scans"}
    }},
    {"$unwind": "$scans_array"},
    {"$group": {
        "_id": "$scans_array.k",
        "detecciones": {
            "$sum": {"$cond": [{"$eq": ["$scans_array.v.detected", True]}, 1, 0]}
        },
        "total_escaneos": {"$sum": 1}
    }},
    {"$project": {
        "detecciones": 1,
        "total_escaneos": 1,
        "ratio": {"$divide": ["$detecciones", "$total_escaneos"]}
    }},
    {"$sort": {"detecciones": -1}}
]

# Resultado: Un documento por cada motor de análisis, mostrando el número de detecciones y el total de escaneos realizados por ese motor.
for result in collection.aggregate(agg_por_motor):
    print(f"Motor: {result['_id']:<25}, Detecciones: {result['detecciones']:<15}, Total escaneos: {result['total_escaneos']:<15}, Ratio: {result['ratio']:.4f}")
    