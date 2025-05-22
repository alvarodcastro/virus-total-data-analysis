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

# Ruta para guardar los resultados
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')

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
world_map.save(RESULTS_DIR + "/mapa_muestras_por_pais.html")
print("Mapa guardado en: mapa_muestras_por_pais.html")

# 4. Crear gráfico de barras
plt.figure(figsize=(12, 6))
plt.bar(df['country'], df['count'], color='skyblue')
plt.xlabel('País')
plt.ylabel('Número de muestras')
plt.title('Número de muestras por país')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(RESULTS_DIR + "/muestras_por_pais.png")
# plt.show()

# GUardar los resultados en un archivo CSV
df.to_csv(RESULTS_DIR + "/muestras_por_pais.csv", index=False)

# ======= ANÁLISIS 2: Muestras más detectadas =======
print("\nANALISIS 2:  Muestras más detectadas")
# Este análisis identifica las 10 muestras con más detecciones absolutas (positives), es decir, aquellas que más motores antivirus clasificaron como maliciosas.

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

# Guardar los resultados en un archivo CSV
df_top_muestras = pd.DataFrame(list(collection.aggregate(agg_top_muestras)))
df_top_muestras.to_csv(RESULTS_DIR + "/top_muestras.csv", index=False)

# ======= ANÁLISIS 3: Detección media por motor de análisis =======
print("\nANALISIS 3: Detección media por motor de análisis")
# Este análisis calcula la tasa de detección media de cada motor antivirus, indicando cuáles tienen mayor eficacia relativa.

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

# Guardar los resultados en un archivo CSV
df_por_motor = pd.DataFrame(list(collection.aggregate(agg_por_motor)))
df_por_motor.to_csv(RESULTS_DIR + "/detecciones_por_motor.csv", index=False)

# ======= ANÁLISIS 4: Muestras más peligrosas =======
print("\nANALISIS 4: Muestras más peligrosas")
# 1. $match: Filtra los documentos para incluir solo aquellos donde el campo total es mayor que 0.
# 2. $project: crea un nuevo documento para cada entrada, seleccionando y calculando campos específicos:
# "sha256": 1 Incluye el campo sha256 en la salida.
# "positives": 1 Incluye el campo positives (probablemente el número de detecciones positivas).
# "total": 1 Incluye el campo total (posiblemente el total de motores de análisis).
# "detection_rate": {"$divide": ["$positives", "$total"]} Crea un nuevo campo llamado detection_rate que es el resultado de dividir positives entre total (proporción de positivos respecto al total).
# 3. $sort: ordena los documentos resultantes de mayor a menor según el campo detection_rate.
# {"detection_rate": -1} Ordena los documentos resultantes de mayor a menor según el campo detection_rate. Así, los archivos con más detecciones positivas respecto al total aparecen primero.  
# 4. $limit: {"$limit": 15} Limita la salida a solo los 15 primeros documentos, es decir, los 15 archivos con más detecciones positivas respecto al total.

agg_top_peligrosas = [
    {"$match": {"total": {"$gt": 0}}},
    {"$project": {
        "sha256": 1,
        "positives": 1,
        "total": 1,
        "detection_rate": {"$divide": ["$positives", "$total"]}
    }},
    {"$sort": {"detection_rate": -1}},
    # {"$limit": 15}
]

# Ejecutar la consulta
top_peligrosas = list(collection.aggregate(agg_top_peligrosas))

# Formatear resultados
tabla = [
    {
        "SHA256": doc["sha256"][:10] + "...",  # acortado para que no desborde
        "Detecciones": doc["positives"],
        "Total AV": doc["total"],
        "Ratio": round(doc["detection_rate"], 4)
    }
    for doc in top_peligrosas
]
# Resultado: Selecciona los 15 archivos con más detecciones positivas respecto al total, mostrando su hash (sha256), el número de positivos, el total de motores y la proporción de positivos respecto al total.

# Mostrar resultados
for fila in tabla:
    print(f"SHA256: {fila['SHA256']}, Detecciones: {fila['Detecciones']:<10}, Total AV: {fila['Total AV']:<10}, Ratio: {fila['Ratio']}")

# Guardar los resultados en un archivo CSV
df_top_peligrosas = pd.DataFrame(tabla)
df_top_peligrosas.to_csv(RESULTS_DIR + "/top_muestras_peligrosas.csv", index=False)

# ANÁLISIS 5:
print("\nANALISIS 5: Muestras con más hijos")
# Agregación MongoDB para contar y clasificar los children por tipo
agg_children_por_muestra = [
    {
        "$match": {
            "children": {"$type": "array"}
        }
    },
    {
        "$project": {
            "sha256": 1,
            "num_children": {"$size": "$children"},
            "children_types": {
                "$arrayToObject": {
                    "$map": {
                        "input": {"$setUnion": [{"$map": {
                            "input": "$children",
                            "as": "child",
                            "in": "$$child.type"
                        }}]},
                        "as": "tipo",
                        "in": {
                            "k": "$$tipo",
                            "v": {
                                "$size": {
                                    "$filter": {
                                        "input": "$children",
                                        "as": "c",
                                        "cond": {"$eq": ["$$c.type", "$$tipo"]}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    {
        "$sort": {"num_children": -1}
    },
    {"$limit": 30}  # Limitar a las 20 muestras con más hijos
]

# Guardar los resultados en un archivo CSV
fase1_results = list(collection.aggregate(agg_children_por_muestra))
df_fase1 = pd.DataFrame(fase1_results)  
df_fase1.to_csv(RESULTS_DIR + "/children_por_muestra.csv", index=False)

# Mostrar resultados
for result in fase1_results:
    print(f"SHA256: {result['sha256']}, Nº hijos: {result['num_children']}, Hijos por tipo: {result['children_types']}")


# ANÁLISIS 6: Muestras con hijos compartidos
print("\nANALISIS 6: Muestras con hijos compartidos")
# Fase 2: Agrupar por tipo de children y contar las muestras relacionadas
agg_relaciones_children = [
    {"$match": {"children": {"$type": "array"}}},
    {"$unwind": "$children"},
    {"$group": {
        "_id": "$children.sha256",
        "filename": {"$first": "$children.filename"},  # Opcional, informativo
        "muestras_relacionadas": {"$addToSet": "$sha256"}
    }},
    {"$project": {
        "filename": 1,
        "num_muestras": {"$size": "$muestras_relacionadas"},
        "muestras_relacionadas": 1
    }},
    {"$match": {"num_muestras": {"$gt": 1}}},
    {"$sort": {"num_muestras": -1}},
    {"$limit": 30}
]


fase2_results = list(collection.aggregate(agg_relaciones_children))
df_fase2 = pd.DataFrame(fase2_results)
df_fase2.to_csv(RESULTS_DIR + "/children_compartidos.csv", index=False)

for result in fase2_results:
    short_sha = result['_id'][:10] + "..." if result['_id'] else ""
    short_muestras = [m[:10] + "..." for m in result['muestras_relacionadas']]
    print(f"SHA256 child: {short_sha}, Número de muestras relacionadas: {result['num_muestras']}")#, Muestras relacionadas: {short_muestras}")

# ==== Análisis 7: Distribución de TAGS ====
print("\nANALISIS 7: Distribución de TAGS")
# Este análisis calcula la cantidad de muestras por cada tag y la detección media asociada a cada uno.
agg_tags = [
    {"$match": {"tags": {"$exists": True, "$ne": []}}},
    {"$unwind": "$tags"},
    {"$group": {
        "_id": "$tags",
        "num_muestras": {"$sum": 1},
        "deteccion_media": {"$avg": {"$divide": ["$positives", "$total"]}}
    }},
    {"$sort": {"num_muestras": -1}}
]

tags_result = list(collection.aggregate(agg_tags))
df_tags = pd.DataFrame(tags_result)
df_tags.to_csv(RESULTS_DIR + "/analisis_tags.csv", index=False)

for result in tags_result:
    print(f"TAG: {result['_id']:<20}, Número de muestras: {result['num_muestras']:<20}, Detección media: {result['deteccion_media']}")

# ==== Análisis 8: Evolución temporal de muestras ====
print("\nANALISIS 8: Evolución temporal de muestras")
# Este análisis calcula la evolución temporal de las muestras, agrupando por fecha de envío y contando el número de muestras por día.
agg_evolucion = [
    {"$match": {"submission.date": {"$exists": True}}},

    {"$project": {
        "fecha": {
                "$substr": ["$submission.date", 0, 10]  # Extrae 'YYYY-MM-DD'
        }
    }},
    {"$group": {
        "_id": "$fecha",
        "num_muestras": {"$sum": 1}
    }},
    {"$sort": {"_id": 1}}
]

evolucion_result = list(collection.aggregate(agg_evolucion))
df_evolucion = pd.DataFrame(evolucion_result)
df_evolucion.to_csv(RESULTS_DIR + "/evolucion_muestras.csv", index=False)
# 
for result in evolucion_result:
    print(f"Fecha: {result['_id']}, Número de muestras: {result['num_muestras']}")



print("\nANALISIS 9: Muestras ordenadas por hora de envío")
# Este análisis agrupa las muestras por hora de envío y las ordena cronológicamente.
agg_por_hora = [
    {"$match": {"submission.date": {"$exists": True}}},
    {"$project": {
        "sha256": 1,
        "hora_envio": "$submission.date"  # Guarda la fecha/hora en un campo llamado 'hora_envio'
        }
    },
    {"$sort": {"hora_envio": 1}}
]

por_hora_result = list(collection.aggregate(agg_por_hora))
df_por_hora = pd.DataFrame(por_hora_result)
df_por_hora.to_csv(RESULTS_DIR + "/muestras_por_hora.csv", index=False)

print("\nFechas de 'submission.date' de todos los documentos:")

for result in por_hora_result:
    print(f"SHA256: {result['sha256']}, Hora de envío: {result['hora_envio']}")

    
import folium
import pycountry
from geopy.geocoders import Nominatim
from time import sleep

# === Tu DataFrame 'df' ya debe contener:
# 'country_code', 'country', 'count', 'avg_detection'

# Diccionario aproximado de coordenadas por país (puedes ampliarlo si necesitas más)
coords_dict = {
    'Canada': (56.1304, -106.3468),
    'Germany': (51.1657, 10.4515),
    'United States': (37.0902, -95.7129),
    'Czechia': (49.8175, 15.4730),
    'Korea, Republic of': (35.9078, 127.7669),
    'India': (20.5937, 78.9629),
    'Ireland': (53.4129, -8.2439),
    'Ecuador': (-1.8312, -78.1834),
    'Russian Federation': (61.5240, 105.3188),
    'Argentina': (-38.4161, -63.6167),
    # Añade más si lo deseas
}

# Crear el mapa base
world_map = folium.Map(location=[20, 0], zoom_start=2)

# Choropleth coloreado por detección media
choropleth = folium.Choropleth(
    geo_data='https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json',
    name='Detección media por país',
    data=df,
    columns=['country', 'avg_detection'],
    key_on='feature.properties.name',
    fill_color='YlGnBu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Detección media',
).add_to(world_map)

# Añadir etiquetas en cada país
for idx, row in df.iterrows():
    country = row['country']
    if country in coords_dict:
        lat, lon = coords_dict[country]
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='black',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>{country}</b><br>Muestras: {row['count']}<br>Media detección: {round(row['avg_detection'], 2)}",
                max_width=200
            )
        ).add_to(world_map)

# Guardar el resultado
RESULTS_DIR = './results'
world_map.save(RESULTS_DIR + "/mapa_muestras_con_info.html")
print("✅ Mapa generado: mapa_muestras_con_info.html")