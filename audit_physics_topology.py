"""Print the reproducible topology distance audit used by PHYSICS_EVIDENCE.md."""
from __future__ import annotations

import json
import math

from topology import CITY_COORDS, CORRIDORS, build_topology, shortest_fiber_route


def geodesic_km(a: str, b: str) -> float:
    lat1, lon1 = map(math.radians, CITY_COORDS[a])
    lat2, lon2 = map(math.radians, CITY_COORDS[b])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    haversine = (math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) *
                 math.sin(dlon / 2) ** 2)
    return 6371.0 * 2.0 * math.asin(math.sqrt(haversine))


def audit() -> dict:
    graph = build_topology()
    fiber_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                   if d["link_type"] == "fiber"]
    fso_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                 if d["link_type"] == "fso"]
    detour_lengths = {}
    for _, _, attrs in fso_edges:
        key = tuple(attrs["detour_for"])
        detour_lengths[key] = detour_lengths.get(key, 0.0) + attrs["distance_km"]
    route = shortest_fiber_route(graph)
    corridors = []
    for start, end, hop_count in CORRIDORS:
        assigned = 80.0 * hop_count
        direct = geodesic_km(start, end)
        corridors.append({
            "start": start, "end": end, "fiber_hops": hop_count,
            "assigned_km": assigned, "geodesic_km": direct,
            "assigned_to_geodesic_ratio": assigned / direct,
            "shorter_than_geodesic": assigned + 1e-9 < direct,
        })
    return {
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "fiber_edges": len(fiber_edges), "fso_edges": len(fso_edges),
        "fso_detours": len(detour_lengths),
        "fiber_hop_lengths_km": sorted({d["distance_km"] for _, _, d in fiber_edges}),
        "fso_segment_lengths_km": sorted({d["distance_km"] for _, _, d in fso_edges}),
        "fso_links_per_detour": len(fso_edges) // max(len(detour_lengths), 1),
        "fso_detour_path_lengths_km": sorted(set(detour_lengths.values())),
        "delhi_chennai_fiber_hops": len(route) - 1,
        "delhi_chennai_fiber_km": sum(graph[a][b]["distance_km"]
                                       for a, b in zip(route, route[1:])),
        "corridors": corridors,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
