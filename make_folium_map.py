"""Create an interactive map of the India road-corridor QKD topology.

The shown fiber paths are routed on OpenStreetMap roads as an alignment proxy.
They are not confirmed telecom cable records. FSO relays assume clear LOS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import folium
import networkx as nx

from topology import (CITIES, CITY_COORDS, CORRIDOR_PAIRS, HUB_CITIES,
                      build_topology, shortest_fiber_route)


def _route_from_artifact(route_file: Path) -> list[str]:
    entries: list[dict[str, Any]] = json.loads(route_file.read_text(encoding="utf-8"))
    policies = [entry for entry in entries if entry.get("model") == "GNN" and entry.get("policy")]
    if not policies:
        raise ValueError(f"No GNN policy route found in {route_file}")
    policy = policies[-1]["policy"]
    return policy if isinstance(policy, list) else policy["sample_path"]


def _corridor_path(graph: nx.Graph, pair: tuple[str, str]) -> list[str]:
    start, end = pair
    route_id = f"{start}--{end}"
    if route_id not in {data.get("corridor_id") for _, _, data in graph.edges(data=True)}:
        route_id = f"{end}--{start}"
    sub = nx.Graph()
    sub.add_edges_from((u, v, data) for u, v, data in graph.edges(data=True)
                       if data.get("corridor_id") == route_id and data["link_type"] == "fiber")
    return nx.shortest_path(sub, start, end, weight="distance_km")


def _path_geometry(graph: nx.Graph, path: list[str]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for u, v in zip(path, path[1:]):
        attrs = graph[u][v]
        if attrs["link_type"] == "fiber":
            segment = list(attrs["geometry"])
            if u != attrs["geometry_from"]:
                segment.reverse()
        else:
            segment = [graph.nodes[u]["pos"], graph.nodes[v]["pos"]]
        coords.extend(segment if not coords else segment[1:])
    return coords


def create_map(output: Path, route_file: Path | None = None) -> Path:
    graph = build_topology()
    if route_file is not None:
        route = _route_from_artifact(route_file)
        if any(node not in graph for node in route):
            raise ValueError(
                "The supplied route was trained on a different topology. "
                "Retrain the policy or omit --route-file to show the current fiber baseline."
            )
        route_name = "Saved GNN route"
        route_title = "Learned route"
    else:
        route = shortest_fiber_route(graph)
        route_name = "Shortest fiber baseline (not policy output)"
        route_title = "Shortest fiber baseline"

    m = folium.Map(location=[20.5, 80.0], zoom_start=5, tiles=None,
                   control_scale=True, prefer_canvas=True)
    for lat in range(10, 36, 5):
        folium.PolyLine([[lat, 67], [lat, 95]], color="#d5dde3", weight=0.8,
                        opacity=0.85, dash_array="2 5").add_to(m)
    for lon in range(70, 96, 5):
        folium.PolyLine([[7, lon], [34, lon]], color="#d5dde3", weight=0.8,
                        opacity=0.85, dash_array="2 5").add_to(m)

    fiber_layer = folium.FeatureGroup(name="Fiber route proxy (maximum 80 km/span)", show=True)
    fso_layer = folium.FeatureGroup(name="Candidate FSO paths (maximum 10 km/link)", show=False)
    requested_hubs = folium.FeatureGroup(name="Hub connectivity corridors", show=True)
    mumbai_kolkata = folium.FeatureGroup(
        name="Mumbai–Kolkata direct routed corridor", show=True)
    learned_layer = folium.FeatureGroup(name=route_name, show=True)

    for u, v, attrs in graph.edges(data=True):
        points = [graph.nodes[u]["pos"], graph.nodes[v]["pos"]]
        if attrs["link_type"] == "fiber":
            folium.PolyLine(
                attrs["geometry"], color="#667785", weight=2.0, opacity=0.52,
                tooltip=(f"Road-aligned fiber proxy | {attrs['corridor'][0]}–{attrs['corridor'][1]} | "
                         f"{attrs['distance_km']:.1f} km | refs: {', '.join(attrs['road_refs'][:4])}"),
            ).add_to(fiber_layer)
        else:
            folium.PolyLine(
                points, color="#1786b4", weight=1.4, opacity=0.46,
                dash_array="4 4",
                tooltip=(f"Candidate FSO | {attrs['distance_km']:.2f} km | "
                         "clear LOS not verified"),
            ).add_to(fso_layer)

    # Highlight one routed fiber corridor for each requested hub-city pair.
    for hub in HUB_CITIES:
        for city in CITIES:
            if city == hub:
                continue
            pair = tuple(sorted((hub, city)))
            path = _corridor_path(graph, pair)
            folium.PolyLine(
                _path_geometry(graph, path),
                color="#2477a5", weight=3.3, opacity=0.66,
                tooltip=f"{hub}–{city} routed corridor (OSM road proxy; OFC unverified)",
            ).add_to(requested_hubs)

    pair = tuple(sorted(("Mumbai", "Kolkata")))
    path = _corridor_path(graph, pair)
    folium.PolyLine(
        _path_geometry(graph, path),
        color="#a14fa3", weight=5, opacity=0.9,
        tooltip="Mumbai–Kolkata direct routed corridor; proposed fiber alignment, not verified OFC",
    ).add_to(mumbai_kolkata)

    folium.PolyLine(
        _path_geometry(graph, route),
        color="#d1493f", weight=5, opacity=0.95,
        tooltip=f"{route_title} | {len(route) - 1} fiber hops",
    ).add_to(learned_layer)

    fiber_layer.add_to(m)
    fso_layer.add_to(m)
    requested_hubs.add_to(m)
    mumbai_kolkata.add_to(m)
    learned_layer.add_to(m)

    for city in CITIES:
        lat, lon = CITY_COORDS[city]
        color = "#d1493f" if city in (route[0], route[-1]) else "#163a5f"
        folium.CircleMarker([lat, lon], radius=5.5, color="white", weight=1.5,
                            fill=True, fill_color=color, fill_opacity=1,
                            tooltip=f"{city} (approximate city center)").add_to(m)
        folium.Marker([lat, lon], icon=folium.DivIcon(html=(
            '<div style="font:600 12px Arial,sans-serif;color:#172b3a;'
            'text-shadow:0 1px 2px white,1px 0 2px white,-1px 0 2px white;'
            'white-space:nowrap;transform:translate(8px,-7px);">'
            f"{city}</div>"))).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    note = (
        f"{route_title}. Fiber corridors follow fastest OpenStreetMap driving routes as a proxy; "
        "public detailed telecom cable alignments were not available. FSO links assume clear "
        "line of sight and are not terrain-checked."
    )
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;top:12px;left:12px;z-index:9999;background:#fff;'
        'padding:9px 13px;border:1px solid #9aa7b1;border-radius:4px;max-width:610px;'
        'box-shadow:0 1px 4px #777;font:11px Arial,sans-serif">'
        '<b style="font-size:16px">India QKD road-corridor topology</b><br>'
        f'{note} Map data: <a href="https://www.openstreetmap.org/copyright">'
        '© OpenStreetMap contributors</a> (ODbL).</div>'
    ))
    m.get_root().header.add_child(folium.Element(
        '<style>.leaflet-container{background:#f4f7f8!important;}</style>'
    ))
    m.fit_bounds([[7.0, 67.0], [34.0, 95.0]])
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures/india_road_corridor_map.html"))
    parser.add_argument("--route-file", type=Path,
                        help="Optional saved route JSON; must be from the current topology")
    args = parser.parse_args()
    print(create_map(args.output, args.route_file))


if __name__ == "__main__":
    main()
