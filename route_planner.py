"""
route_planner.py
Sistema inteligente de rutas TransMilenio con modelo de predicción supervisado.
"""

import networkx as nx
import joblib
import datetime

# Cargar modelo entrenado
MODEL_PATH = "models/model_travel_time.joblib"
model = joblib.load(MODEL_PATH)

# Crear el grafo de TransMilenio (simplificado)
G = nx.Graph()

# Lista simple de estaciones (puedes ampliarla)
stations = [
    "Portal Norte", "Calle 170", "Calle 145", "Calle 127", "Calle 100",
    "Calle 85", "Calle 72", "U. Nacional", "Calle 57", "Zona T", "Portal Sur"
]

# Crear conexiones básicas (rutas troncales)
edges = [
    ("Portal Norte", "Calle 170"),
    ("Calle 170", "Calle 145"),
    ("Calle 145", "Calle 127"),
    ("Calle 127", "Calle 100"),
    ("Calle 100", "Calle 85"),
    ("Calle 85", "Calle 72"),
    ("Calle 72", "U. Nacional"),
    ("U. Nacional", "Calle 57"),
    ("Calle 57", "Zona T"),
    ("Zona T", "Portal Sur")
]

# Añadir conexiones al grafo
for o, d in edges:
    G.add_edge(o, d)

def predict_time(origin, dest, hour=None, day_of_week=None):
    """
    Usa el modelo ML para predecir el tiempo de viaje entre dos estaciones.
    """
    if hour is None:
        hour = datetime.datetime.now().hour
    if day_of_week is None:
        day_of_week = datetime.datetime.now().weekday()

    # variables base (ajústalas según tu modelo)
    sample = {
        "origin": origin,
        "dest": dest,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_peak": 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0,
        "distance_km": 0.8,
        "stops_apart": abs(stations.index(origin) - stations.index(dest)),
        "num_transfers": 0,
        "origin_line": "A",
        "dest_line": "A"
    }

    # convertir a DataFrame
    import pandas as pd
    df_sample = pd.DataFrame([sample])

    # predecir
    pred = model.predict(df_sample)[0]
    return float(round(pred, 2))

def set_edge_weights(G):
    """
    Asigna pesos (tiempo estimado) a cada conexión del grafo según el modelo.
    """
    for (o, d) in G.edges():
        predicted = predict_time(o, d)
        G[o][d]['weight'] = predicted
    print("Pesos de las aristas actualizados usando el modelo.")

def best_route(origin, dest):
    """
    Encuentra la mejor ruta entre dos estaciones usando los pesos del modelo.
    """
    try:
        path = nx.shortest_path(G, source=origin, target=dest, weight='weight')
        total_time = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        return path, round(total_time, 2)
    except nx.NetworkXNoPath:
        return None, None

if __name__ == "__main__":
    set_edge_weights(G)
    origin = input("Ingrese estación de origen: ").strip()
    dest = input("Ingrese estación de destino: ").strip()

    if origin not in G.nodes or dest not in G.nodes:
        print("Alguna de las estaciones ingresadas no existe.")
    else:
        path, time = best_route(origin, dest)
        if path:
            print(f"\nRuta óptima: {' -> '.join(path)}")
            print(f"Tiempo estimado de viaje: {time} minutos\n")
        else:
            print("No existe una ruta entre las estaciones seleccionadas.")
