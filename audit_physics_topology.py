"""Audit corridor connectivity and fiber/FSO span limits."""
from __future__ import annotations

import json

from topology import (CORRIDOR_METADATA, CORRIDORS, HUB_CITIES, CITIES,
                      FIBER_SPAN_KM, FSO_SPAN_KM, build_topology,
                      shortest_fiber_route, topology_summary)


def audit() -> dict:
    graph = build_topology()
    fiber_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                   if d["link_type"] == "fiber"]
    fso_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                 if d["link_type"] == "fso"]
    route = shortest_fiber_route(graph)
    corridors = []
    for start, end, fiber_hops in CORRIDORS:
        route_id = f"{start}--{end}"
        attrs = [d for _, _, d in fiber_edges if d["corridor_id"] == route_id]
        corridors.append({
            "start": start,
            "end": end,
            "fiber_hops": len(attrs),
            "modeled_fiber_km": sum(d["distance_km"] for d in attrs),
            "osrm_fastest_driving_km": CORRIDOR_METADATA[route_id]["distance_m"] / 1000,
            "road_refs": sorted({ref for d in attrs for ref in d["road_refs"]}),
            "road_reference": "OSM driving-route references; mixed road classes",
        })
    hub_connectivity = {
        hub: sorted(city for city in CITIES if city != hub and
                    graph.has_edge(hub, city) or
                    (city != hub and any(
                        d.get("corridor") == tuple(sorted((hub, city)))
                        for _, _, d in graph.edges(data=True))))
        for hub in HUB_CITIES
    }
    return {
        "summary": topology_summary(graph),
        "source": graph.graph["corridor_source"],
        "retrieved_utc": graph.graph["corridor_retrieved_utc"],
        "fiber_span_limit_km": FIBER_SPAN_KM,
        "max_fiber_span_km": max(d["distance_km"] for _, _, d in fiber_edges),
        "fso_span_limit_km": FSO_SPAN_KM,
        "max_fso_line_distance_km": max(d["distance_km"] for _, _, d in fso_edges),
        "all_fso_visibility_verified": False,
        "hub_city_corridors": hub_connectivity,
        "delhi_chennai_fiber_hops": len(route) - 1,
        "delhi_chennai_fiber_km": sum(graph[a][b]["distance_km"]
                                       for a, b in zip(route, route[1:])),
        "corridors": corridors,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
