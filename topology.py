"""Deterministic seven-city QKD network topology."""
from __future__ import annotations

import math
from typing import Any

import networkx as nx


CITIES = ("Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Jaipur", "Hyderabad")
DEFAULT_SOURCE = "Delhi"
DEFAULT_DESTINATION = "Chennai"

# Corridor hop counts sum to 87. Delhi is a junction, and the Delhi–Jaipur–
# Mumbai–Bangalore–Chennai all-fiber route is 23 hops (the reference route).
CORRIDORS = (
    ("Delhi", "Jaipur", 3),
    ("Jaipur", "Mumbai", 9),
    ("Mumbai", "Bangalore", 6),
    ("Bangalore", "Chennai", 5),
    ("Delhi", "Hyderabad", 16),
    ("Hyderabad", "Chennai", 8),
    ("Delhi", "Kolkata", 17),
    ("Kolkata", "Chennai", 23),
)

# Approximate city coordinates are used only to make the stored topology
# inspectable; the corridor distances below define the simulation's km scale.
CITY_COORDS = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639),
    "Bangalore": (12.9716, 77.5946), "Jaipur": (26.9124, 75.7873),
    "Hyderabad": (17.3850, 78.4867),
}


def _interpolate(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def build_topology() -> nx.Graph:
    """Return a connected graph with 87 fiber links and 232 short FSO links.

    Every corridor fiber hop is 80 km. Each third hop receives an alternate
    eight-link FSO chain, with each segment 10 km. FSO relay nodes are appended
    while iterating corridors, preserving the corridor-by-corridor sequence
    signal intentionally visible to the graph-blind LSTM.
    """
    graph = nx.Graph()
    for city in CITIES:
        graph.add_node(city, node_type="city", city=True, pos=CITY_COORDS[city])

    relay_index = 0
    corridor_index = 0
    backbone_index = 0
    for start, end, hop_count in CORRIDORS:
        corridor_index += 1
        path = [start]
        for hop in range(1, hop_count):
            relay_index += 1
            node = f"R{relay_index}"
            # Deliberate trust placement: every fourth backbone relay is full_tn;
            # the others are stn, giving regular but non-dense trust resets.
            node_type = "full_tn" if relay_index % 4 == 0 else "stn"
            pos = _interpolate(CITY_COORDS[start], CITY_COORDS[end], hop / hop_count)
            graph.add_node(node, node_type=node_type, city=False, pos=pos,
                           relay_index=relay_index, corridor_index=corridor_index)
            path.append(node)
        path.append(end)
        for hop, (u, v) in enumerate(zip(path, path[1:]), start=1):
            backbone_index += 1
            graph.add_edge(u, v, link_type="fiber", distance_km=80.0,
                           backbone=True, corridor=(start, end), hop_index=hop)
            if backbone_index % 3 == 0:
                # An alternate 80 km chain split into 8 short 10 km FSO links.
                fso_path = [u]
                p0, p1 = graph.nodes[u]["pos"], graph.nodes[v]["pos"]
                for segment in range(1, 8):
                    relay_index += 1
                    node = f"R{relay_index}"
                    # Alternating lateral offset makes it a detour while keeping
                    # each modeled link's explicit length at the target 10 km.
                    base = _interpolate(p0, p1, segment / 8)
                    offset = 0.012 if segment % 2 else -0.012
                    pos = (base[0] + offset, base[1] + offset)
                    node_type = "full_tn" if relay_index % 4 == 0 else "stn"
                    graph.add_node(node, node_type=node_type, city=False, pos=pos,
                                   relay_index=relay_index, corridor_index=corridor_index)
                    fso_path.append(node)
                fso_path.append(v)
                for segment, (a, b) in enumerate(zip(fso_path, fso_path[1:]), start=1):
                    graph.add_edge(a, b, link_type="fso", distance_km=10.0,
                                   backbone=False, detour_for=(u, v),
                                   corridor=(start, end), segment_index=segment)

    graph.graph.update(
        seasons=("normal", "summer", "winter", "monsoon"),
        source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION,
        fiber_outage_prob=0.01,
    )
    return graph


def shortest_fiber_route(graph: nx.Graph, source: str = DEFAULT_SOURCE,
                         destination: str = DEFAULT_DESTINATION) -> list[str]:
    """Dijkstra-km route restricted to the fiber backbone."""
    fiber = nx.Graph()
    fiber.add_nodes_from((n, d) for n, d in graph.nodes(data=True))
    fiber.add_edges_from((u, v, d) for u, v, d in graph.edges(data=True)
                         if d["link_type"] == "fiber")
    return nx.shortest_path(fiber, source, destination, weight="distance_km")


def topology_summary(graph: nx.Graph) -> dict[str, Any]:
    fiber_count = sum(d["link_type"] == "fiber" for _, _, d in graph.edges(data=True))
    fso_count = graph.number_of_edges() - fiber_count
    route = shortest_fiber_route(graph)
    return {
        "nodes": graph.number_of_nodes(), "links": graph.number_of_edges(),
        "fiber_links": fiber_count, "fso_links": fso_count,
        "default_route_hops": len(route) - 1,
        "default_route_km": sum(graph[a][b]["distance_km"] for a, b in zip(route, route[1:])),
    }


if __name__ == "__main__":
    g = build_topology()
    print(topology_summary(g))
