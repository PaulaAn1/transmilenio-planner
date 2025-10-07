# generate_and_train.py
"""
Genera un dataset sintético de viajes en Transmilenio y entrena un modelo supervisado
para predecir el tiempo de viaje (travel_time_min).
Salida:
 - transmilenio_trips_sample.csv
 - model_travel_time.joblib
 - metrics.txt
 - example_runs.txt (ejemplo de predicciones)
"""

import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 1) Generar dataset sintético
def generate_synthetic(n_samples=2000):
    # Lista de estaciones simplificada (puedes ampliarla)
    estaciones = [
        "Portal Norte","Calle 170","Calle 145","Calle 127","Calle 100",
        "Calle 85","Calle 72","U. Nacional","Zona T","Calle 57","Portal Sur"
    ]
    lines = ["A","B","C","D"]
    data = []
    for _ in range(n_samples):
        origin = random.choice(estaciones)
        dest = random.choice(estaciones)
        # evitar origen==dest en la muestra (o permitir y setear tiempo=0)
        while dest == origin:
            dest = random.choice(estaciones)
        hour = random.randint(4, 23)  # servicio 4am-23pm
        day_of_week = random.randint(0,6)  # 0=Monday
        is_peak = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        # Estimar 'distance_km' en función de posiciones ficticias:
        idx_o = estaciones.index(origin)
        idx_d = estaciones.index(dest)
        stops_apart = abs(idx_o - idx_d)
        distance_km = round(0.8 * stops_apart + np.random.normal(0, 0.5), 2)
        num_transfers = 0
        # simular transbordos probabilísticamente si muchas paradas
        if stops_apart > 3 and random.random() < 0.4:
            num_transfers = random.choice([1,2])
        # base time: per stop ~4.5 min, add transfer penalty and peak penalty + noise
        base_time = stops_apart * 4.5 + num_transfers * 5 + is_peak * 3
        # add random congestion/noise
        travel_time = max(2.0, round(base_time + np.random.normal(0, 3.0), 2))
        # pick lines for origin and destination (for modeling)
        origin_line = random.choice(lines)
        dest_line = random.choice(lines)
        data.append({
            "origin": origin,
            "dest": dest,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_peak": is_peak,
            "distance_km": max(0.1, distance_km),
            "stops_apart": stops_apart,
            "num_transfers": num_transfers,
            "origin_line": origin_line,
            "dest_line": dest_line,
            "travel_time_min": travel_time
        })
    df = pd.DataFrame(data)
    return df

# 2) Guardar CSV
def save_dataset(df, path="transmilenio_trips_sample.csv"):
    df.to_csv(path, index=False)
    print(f"Dataset guardado en {path} ({len(df)} filas)")

# 3) Entrenar modelo supervisado (regresor)
def train_model(df, model_path="model_travel_time.joblib"):
    # Features y target
    X = df.drop(columns=["travel_time_min"])
    y = df["travel_time_min"]

    # columnas categóricas y numéricas
    cat_cols = ["origin","dest","origin_line","dest_line","day_of_week","hour","is_peak"]
    num_cols = ["distance_km","stops_apart","num_transfers"]

    # OneHotEncoder compatible con versiones viejas y nuevas
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer([
        ("cat", encoder, cat_cols),
    ], remainder="passthrough")

    # Pipeline
    pipe = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1))
    ])

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED)

    # entrenar
    pipe.fit(X_train, y_train)

    # predecir y evaluar
    y_pred_train = pipe.predict(X_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    # RMSE: usar squared=False si está disponible, si no calcular sqrt(MSE)
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(sqrt(mse))

    r2 = r2_score(y_test, y_pred)

    # guardar modelo
    joblib.dump(pipe, model_path)

    # report
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

    print("Entrenamiento finalizado. Métricas:", metrics)
    return pipe, metrics, X_test, y_test, y_pred

# 4) Guardar report y ejemplo
def save_reports(metrics, sample_inference, example_txt="example_runs.txt", metrics_txt="metrics.txt"):
    with open(metrics_txt, "w") as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v}\n")
    with open(example_txt, "w") as f:
        f.write("Ejemplo de predicciones:\n")
        for inp, pred in sample_inference:
            f.write(f"Input: {inp}\nPredicción (min): {pred}\n\n")
    print(f"Reports guardados: {metrics_txt}, {example_txt}")

def main():
    # generar
    df = generate_synthetic(2200)
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "transmilenio_trips_sample.csv")
    save_dataset(df, csv_path)

    # entrenar
    model_path = os.path.join("models", "model_travel_time.joblib")
    os.makedirs("models", exist_ok=True)
    pipe, metrics, X_test, y_test, y_pred = train_model(df, model_path=model_path)

    # crear ejemplos de inferencia
    sample_inp = X_test.sample(6, random_state=RANDOM_SEED).to_dict(orient="records")
    sample_preds = pipe.predict(pd.DataFrame(sample_inp))
    sample_pairs = list(zip(sample_inp, [float(round(p,2)) for p in sample_preds]))

    save_reports(metrics, sample_pairs)

if __name__ == "__main__":
    main()
