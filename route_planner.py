#!/usr/bin/env python3
"""
route_planner.py

Sistema inteligente simple que:
- usa una base de conocimiento (reglas lógicas en un formato legible)
- construye un grafo de estados (estación, línea) considerando recorridos por línea y transbordos del sistema masivo de trasporte de Transmilenio
- busca la ruta de menor tiempo entre punto A y punto B (minimizando tiempo total)
"""

from heapq import heappush, heappop
from typing import Dict, List, Tuple, Set

# ----------------------------
# BASE DE CONOCIMIENTO (REGLAS)
# ----------------------------
# Definimos la base como listas/tuplas. cada "línea" tiene un nombre y una lista ordenada de estaciones
# También definimos tiempos de viaje entre estaciones consecutivas en la misma línea.
# Transferencias (cambio de línea) en la misma estación tienen un tiempo de transbordo.

# Ejemplo reducido
LINES = {
    "A": ["Portal Norte", "Calle 170", "Calle 145", "Calle 127", "Calle 100"],
    "B": ["Calle 100", "Calle 85", "Calle 72", "U. Nacional"],
    "C": ["U. Nacional", "Zona T", "Calle 57", "Portal Sur"]
}

# Tiempo en minutos entre estaciones consecutivas por línea (puede variar por tramo)
# Si no está explícito, se asume un tiempo por defecto.
TRAVEL_TIME_BETWEEN = {
    # (line, station_i, station_j): minutes
    ("A", "Portal Norte", "Calle 170"): 6,
    ("A", "Calle 170", "Calle 145"): 4,
    ("A", "Calle 145", "Calle 127"): 3,
    ("A", "Calle 127", "Calle 100"): 4,
    ("B", "Calle 100", "Calle 85"): 5,
    ("B", "Calle 85", "Calle 72"): 4,
    ("B", "Calle 72", "U. Nacional"): 6,
    ("C", "U. Nacional", "Zona T"): 5,
    ("C", "Zona T", "Calle 57"): 4,
    ("C", "Calle 57", "Portal Sur"): 8,
}

DEFAULT_TRAVEL = 5  # minutos por tramo si no está especificado

TRANSFER_TIME = 4  # tiempo promedio de transbordo en minutos (misma estación, otra línea)
WALK_TIME_NEIGHBOR = 10  # caminar entre estaciones distintas (no usado por defecto)


# ----------------------------
# CONSTRUCCIÓN DEL GRAFO
# ----------------------------
# Nodo: (station, line)
# Arista entre (S_i, L) y (S_j, L) para estaciones consecutivas de L -> peso = travel_time
# Arista entre (S, L1) y (S, L2) si misma estación sirve ambas líneas -> peso = TRANSFER_TIME

def build_graph(lines: Dict[str, List[str]]) -> Tuple[Dict[Tuple[str, str], List[Tuple[Tuple[str, str], int]]], Set[Tuple[str,str]]]:
    graph: Dict[Tuple[str, str], List[Tuple[Tuple[str, str], int]]] = {}
    nodes: Set[Tuple[str,str]] = set()

    # nodos por (station, line)
    for line, stations in lines.items():
        for s in stations:
            nodes.add((s, line))
            graph.setdefault((s, line), [])

    # aristas a lo largo de la línea (ambos sentidos)
    for line, stations in lines.items():
        for i in range(len(stations) - 1):
            s1 = stations[i]
            s2 = stations[i+1]
            t = TRAVEL_TIME_BETWEEN.get((line, s1, s2), TRAVEL_TIME_BETWEEN.get((line, s2, s1), DEFAULT_TRAVEL))
            graph[(s1, line)].append(((s2, line), t))
            graph[(s2, line)].append(((s1, line), t))

    # transferencias en la misma estación entre líneas
    # si una estación aparece en más de una línea, conectar sus (station,line) con costo TRANSFER_TIME
    station_to_lines: Dict[str, List[str]] = {}
    for line, stations in lines.items():
        for s in stations:
            station_to_lines.setdefault(s, []).append(line)

    for station, line_list in station_to_lines.items():
        if len(line_list) > 1:
            for i in range(len(line_list)):
                for j in range(i+1, len(line_list)):
                    l1 = line_list[i]
                    l2 = line_list[j]
                    graph[(station, l1)].append(((station, l2), TRANSFER_TIME))
                    graph[(station, l2)].append(((station, l1), TRANSFER_TIME))

    return graph, nodes


# ----------------------------
# ALGORITMO DE BÚSQUEDA (Dijkstra con estado (estación, línea))
# ----------------------------
def dijkstra_multi_source(graph, sources: List[Tuple[str,str]], targets: Set[Tuple[str,str]]):
    # standard Dijkstra: nodos = (station, line)
    INF = 10**9
    dist = {}
    prev = {}
    pq = []

    for s in graph.keys():
        dist[s] = INF
        prev[s] = None

    for src in sources:
        if src not in graph:
            continue
        dist[src] = 0
        heappush(pq, (0, src))

    visited = set()
    found_target = None

    while pq:
        d, u = heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u in targets:
            found_target = u
            break

        for (v, w) in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heappush(pq, (nd, v))

    return dist, prev, found_target


def reconstruct_path(prev, end):
    if end is None:
        return []
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


# ----------------------------
# INTERFAZ: encontrar ruta A -> B
# ----------------------------
def find_best_route(lines, A: str, B: str):
    graph, nodes = build_graph(lines)

    # fuentes: todas las (A, line) para líneas que sirven A
    sources = [(A, line) for line, stations in lines.items() if A in stations]
    # objetivos: todas las (B, line)
    targets = set((B, line) for line, stations in lines.items() if B in stations)

    if not sources:
        raise ValueError(f"No hay líneas que sirvan la estación origen: {A}")
    if not targets:
        raise ValueError(f"No hay líneas que sirvan la estación destino: {B}")

    dist, prev, found_target = dijkstra_multi_source(graph, sources, targets)

    if found_target is None:
        return None, None, None  # sin ruta

    path_nodes = reconstruct_path(prev, found_target)
    total_time = dist[found_target]

    # construir instrucciones legibles
    instructions = []
    for i in range(len(path_nodes)-1):
        (station1, line1) = path_nodes[i]
        (station2, line2) = path_nodes[i+1]
        # si se mantiene la línea -> viaje en línea
        if line1 == line2:
            instructions.append(f"Tomar línea {line1} desde '{station1}' hasta '{station2}'")
        else:
            instructions.append(f"En '{station1}' cambiar de la línea {line1} a la línea {line2} (transbordo)")
    return path_nodes, instructions, total_time


# ----------------------------
# EJEMPLO DE USO (puedes cambiar A y B)
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Route planner (Transmilenio ejemplo)")
    parser.add_argument("--from", dest="A", required=False, help="Estación origen", default="Portal Norte")
    parser.add_argument("--to", dest="B", required=False, help="Estación destino", default="Portal Sur")
    args = parser.parse_args()

    A = args.A
    B = args.B

    try:
        path_nodes, instructions, total_time = find_best_route(LINES, A, B)
        if path_nodes is None:
            print(f"No se encontró ruta entre {A} y {B}.")
        else:
            print(f"Ruta óptima de '{A}' a '{B}' (tiempo estimado: {total_time} minutos):\n")
            # mostrar camino detallado
            print(" -> ".join([f"{s}({l})" for (s,l) in path_nodes]))
            print("\nInstrucciones:")
            for step in instructions:
                print(" -", step)
    except ValueError as e:
        print("Error:", e)
