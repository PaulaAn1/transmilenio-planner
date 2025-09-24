# transmilenio-planner

Sistema inteligente de rutas para Transmilenio en Bogotá (Proyecto académico)

# 🚌 Transmilenio Route Planner (Proyecto Académico)

Este proyecto implementa un sistema inteligente que, a partir de una **base de conocimiento expresada en reglas lógicas** (líneas y estaciones de Transmilenio), construye un **grafo de estados (estación + línea)** y encuentra la **ruta óptima** entre dos puntos usando el algoritmo de **Dijkstra**.

---

## 📌 Objetivos

- Representar el sistema de transporte Transmilenio en una **base de conocimiento**.
- Convertir esa base en un grafo considerando **tiempos de viaje** y **transbordos**.
- Desarrollar un planificador que encuentre la mejor ruta (minimizando tiempo).
- Mostrar resultados en formato legible con **instrucciones paso a paso**.

---

## ⚙️ Archivos principales

- **`route_planner.py`** → Código principal en Python.
- **`example_runs.txt`** → Ejemplos de ejecución con salidas del sistema.
- **`video/transmilenio_demo.mp4`** → Video explicativo del proyecto (2–4 minutos).

---

## 🚀 Uso

1. Clona el repositorio:
   ```bash
   git clone https://github.com/PaulaAn1/transmilenio-planner.git
   cd transmilenio-planner
   ```
