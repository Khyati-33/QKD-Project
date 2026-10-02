"""India city network built on stored OpenStreetMap driving-route geometry.

The road paths are an NH/major-road alignment proxy for possible fiber routes,
not mapped telecom assets. FSO segments are hypothetical clear-LOS alternatives
placed along those paths and require later terrain/site validation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx


CITIES = ("Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Jaipur", "Hyderabad")
DEFAULT_SOURCE = "Delhi"
DEFAULT_DESTINATION = "Chennai"
FIBER_SPAN_KM = 80.0
FSO_SPAN_KM = 10.0
FSO_PLACEMENT_STEP_KM = 9.9
HUB_CITIES = ("Jaipur", "Hyderabad", "Kolkata")

# Approximate city-center coordinates are graph endpoints, not surveyed POPs.
CITY_COORDS = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639),
    "Bangalore": (12.9716, 77.5946), "Jaipur": (26.9124, 75.7873),
    "Hyderabad": (17.3850, 78.4867),
}

_ROUTE_FILE = Path(__file__).resolve().parent / "data" / "osm_road_corridors.json"
_ROUTE_DATA: dict[str, Any] = json.loads(_ROUTE_FILE.read_text(encoding="utf-8"))
_ROUTES: dict[str, dict[str, Any]] = _ROUTE_DATA["routes"]
CORRIDOR_METADATA = _ROUTES
CORRIDOR_PAIRS = tuple(
    (record["start"], record["end"])
    for record in _ROUTES.values()
)
# Compatibility summary: (city A, city B, number of <=80 km fiber spans).
CORRIDORS = tuple(
    (record["start"], record["end"], math.ceil(record["distance_m"] / 1000 / FIBER_SPAN_KM))
    for record in _ROUTES.values()
)


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))


def _sample_chainage(points: list[tuple[float, float]], cumulative: list[float],
                     chainage_km: float) -> tuple[float, float]:
    if chainage_km <= 0:
        return points[0]
    if chainage_km >= cumulative[-1]:
        return points[-1]
    lo, hi = 0, len(cumulative) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cumulative[mid] < chainage_km:
            lo = mid
        else:
            hi = mid
    span = cumulative[hi] - cumulative[lo]
    t = 0.0 if span == 0 else (chainage_km - cumulative[lo]) / span
    return (points[lo][0] + t * (points[hi][0] - points[lo][0]),
            points[lo][1] + t * (points[hi][1] - points[lo][1]))


def _section_geometry(points: list[tuple[float, float]], cumulative: list[float],
                      start_km: float, end_km: float) -> list[tuple[float, float]]:
    section = [_sample_chainage(points, cumulative, start_km)]
    section.extend(points[i] for i, chainage in enumerate(cumulative)
                   if start_km < chainage < end_km)
    section.append(_sample_chainage(points, cumulative, end_km))
    return section


def _route_polyline(record: dict[str, Any]) -> tuple[list[tuple[float, float]], list[float]]:
    start, end = record["start"], record["end"]
    road_points = [(lat, lon) for lon, lat in record["coordinates_lon_lat"]]
    # Add short urban-access connectors from city-center model nodes to the
    # nearest road points returned by the router.
    points = [CITY_COORDS[start], *road_points, CITY_COORDS[end]]
    compact = [points[0]]
    for point in points[1:]:
        if _haversine_km(compact[-1], point) > 0.001:
            compact.append(point)
    cumulative = [0.0]
    for a, b in zip(compact, compact[1:]):
        cumulative.append(cumulative[-1] + _haversine_km(a, b))
    return compact, cumulative


def _add_relay(graph: nx.Graph, name: str, pos: tuple[float, float], index: int,
               *, technology: str, node_type: str) -> None:
    graph.add_node(name, node_type=node_type,
                   city=False, pos=pos, relay_index=index, technology=technology)


def build_topology() -> nx.Graph:
    """Build fiber corridors along routed roads, with 10 km FSO alternatives.

    Hub connectivity is explicit: Jaipur, Hyderabad, and Kolkata each have a
    routed corridor to every other configured city. Mumbai therefore has
    direct modeled corridors to Hyderabad and Kolkata. Routes use stored OSM
    driving geometry; they are not evidence of actual OFC placement.
    """
    graph = nx.Graph()
    for city in CITIES:
        graph.add_node(city, node_type="full_tn", city=True, pos=CITY_COORDS[city])

    relay_index = 0
    for route_index, (route_id, record) in enumerate(_ROUTES.items(), start=1):
        start, end = record["start"], record["end"]
        points, cumulative = _route_polyline(record)
        total_km = cumulative[-1]
        if total_km <= 0:
            raise ValueError(f"Empty road corridor {route_id}")
        fiber_chainages = [0.0]
        fiber_chainages.extend(float(km) for km in range(
            int(FIBER_SPAN_KM), math.floor(total_km), int(FIBER_SPAN_KM)))
        if total_km - fiber_chainages[-1] < 1e-6:
            fiber_chainages[-1] = total_km
        else:
            fiber_chainages.append(total_km)

        fiber_nodes = [start]
        for span_index, chainage in enumerate(fiber_chainages[1:-1], start=1):
            relay_index += 1
            name = f"{route_id}_F{span_index}"
            _add_relay(graph, name, _sample_chainage(points, cumulative, chainage),
                       relay_index, technology="fiber_pop", node_type="full_tn")
            fiber_nodes.append(name)
        fiber_nodes.append(end)

        refs = sorted({step["ref"] for step in record.get("road_steps", []) if step.get("ref")})
        for span_index, (u, v) in enumerate(zip(fiber_nodes, fiber_nodes[1:])):
            start_km, end_km = fiber_chainages[span_index:span_index + 2]
            span_km = end_km - start_km
            graph.add_edge(
                u, v, link_type="fiber", distance_km=span_km,
                backbone=True, corridor=(start, end), corridor_id=route_id,
                route_index=route_index, span_index=span_index + 1,
                geometry=_section_geometry(points, cumulative, start_km, end_km),
                geometry_from=u,
                road_source="OpenStreetMap driving route", road_refs=refs,
            )

            detour_id = frozenset((u, v))
            fso_chainages = [start_km]
            cursor = start_km + FSO_PLACEMENT_STEP_KM
            while cursor < end_km - 1e-6:
                fso_chainages.append(cursor)
                cursor += FSO_PLACEMENT_STEP_KM
            if len(fso_chainages) == 1:
                fso_chainages.append((start_km + end_km) / 2)
            fso_chainages.append(end_km)
            fso_nodes = [u]
            for segment_index, chainage in enumerate(fso_chainages[1:-1], start=1):
                relay_index += 1
                name = f"{route_id}_D{span_index + 1}_F{segment_index}"
                _add_relay(graph, name, _sample_chainage(points, cumulative, chainage),
                           relay_index, technology="candidate_fso_relay",
                           node_type=("full_tn" if segment_index % 4 == 0 else "stn"))
                fso_nodes.append(name)
            fso_nodes.append(v)
            for segment_index, (a, b) in enumerate(zip(fso_nodes, fso_nodes[1:]), start=1):
                distance_km = _haversine_km(graph.nodes[a]["pos"], graph.nodes[b]["pos"])
                if distance_km > FSO_SPAN_KM + 1e-6:
                    raise ValueError(f"FSO hop {a}--{b} exceeds {FSO_SPAN_KM} km: {distance_km:.3f}")
                graph.add_edge(
                    a, b, link_type="fso", distance_km=distance_km,
                    backbone=False, detour_for=detour_id,
                    corridor=(start, end), corridor_id=route_id,
                    span_index=span_index + 1, segment_index=segment_index,
                    outage_reference_distance_km=FSO_SPAN_KM,
                    visibility_assumption="unverified_clear_line_of_sight",
                )

    graph.graph.update(
        seasons=("normal", "summer", "winter", "monsoon"),
        source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION,
        fiber_outage_prob=0.01,
        corridor_source=_ROUTE_DATA["source"],
        corridor_retrieved_utc=_ROUTE_DATA["retrieved_utc"],
        corridor_method=_ROUTE_DATA["method"],
        fiber_span_km=FIBER_SPAN_KM,
        fso_span_km=FSO_SPAN_KM,
    )
    return graph


def shortest_fiber_route(graph: nx.Graph, source: str = DEFAULT_SOURCE,
                         destination: str = DEFAULT_DESTINATION) -> list[str]:
    """Shortest road-distance route restricted to modeled fiber links."""
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
        "city_corridors": len(CORRIDORS),
        "default_route_hops": len(route) - 1,
        "default_route_km": sum(graph[a][b]["distance_km"] for a, b in zip(route, route[1:])),
    }


if __name__ == "__main__":
    g = build_topology()
    print(topology_summary(g))
