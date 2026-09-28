"""Create an interactive geographic view of a saved QKD experiment route.

Example:
    python make_folium_map.py --run-dir experiments/runs/<run-name>

The coordinates come from the synthetic topology in ``topology.py``. Corridor
relays are linearly interpolated between city coordinates, so the figure is an
illustrative geographic schematic rather than a surveyed network map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import folium
import networkx as nx

from topology import CITIES, CITY_COORDS, build_topology


def _route_from_artifact(route_file: Path) -> list[str]:
    entries: list[dict[str, Any]] = json.loads(route_file.read_text(encoding="utf-8"))
    policies = [entry for entry in entries if entry.get("model") == "GNN" and entry.get("policy")]
    if not policies:
        raise ValueError(f"No GNN policy route found in {route_file}")
    policy = policies[-1]["policy"]
    # Current evaluation artifacts store the route as a list; support older
    # summaries which nested it under a sample_path key as well.
    return policy if isinstance(policy, list) else policy["sample_path"]


def create_map(run_dir: Path, output: Path) -> Path:
    graph = build_topology()
    routes_file = run_dir / "routes" / "representative_routes.json"
    route = _route_from_artifact(routes_file)
    missing = [node for node in route if node not in graph]
    if missing:
        raise ValueError(f"Route contains nodes absent from topology: {missing}")

    center = [20.5, 80.0]
    m = folium.Map(
        location=center,
        zoom_start=5,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    # A coordinate graticule keeps the map geographically readable without
    # requesting any third-party basemap or tile API.
    for lat in range(10, 36, 5):
        folium.PolyLine([[lat, 67], [lat, 94]], color="#cbd5dd", weight=0.8,
                        opacity=0.8, dash_array="2 5",
                        tooltip=f"{lat}° N").add_to(m)
    for lon in range(70, 96, 5):
        folium.PolyLine([[7, lon], [33, lon]], color="#cbd5dd", weight=0.8,
                        opacity=0.8, dash_array="2 5",
                        tooltip=f"{lon}° E").add_to(m)

    fiber_layer = folium.FeatureGroup(name="Modeled fiber backbone (80 km/link)", show=True)
    fso_layer = folium.FeatureGroup(name="Modeled FSO detours (10 km/link)", show=True)
    route_layer = folium.FeatureGroup(name="Epoch 50 learned route", show=True)
    connectivity_layer = folium.FeatureGroup(
        name="Mumbai–Kolkata connection (via Jaipur and Delhi)", show=True)

    for u, v, attrs in graph.edges(data=True):
        points = [graph.nodes[u]["pos"], graph.nodes[v]["pos"]]
        if attrs["link_type"] == "fiber":
            folium.PolyLine(
                points,
                color="#73808c",
                weight=2.2,
                opacity=0.58,
                tooltip=f"Fiber | modeled {attrs['distance_km']:.0f} km | {attrs['corridor'][0]}–{attrs['corridor'][1]}",
            ).add_to(fiber_layer)
        else:
            folium.PolyLine(
                points,
                color="#1885bd",
                weight=1.7,
                opacity=0.55,
                dash_array="5 5",
                tooltip=f"FSO | modeled {attrs['distance_km']:.0f} km | synthetic detour",
            ).add_to(fso_layer)

    # The model has no direct Mumbai–Kolkata corridor. Show its shortest
    # existing fiber connection explicitly so the network continuity is clear.
    fiber_graph = nx.Graph()
    fiber_graph.add_nodes_from(graph.nodes(data=True))
    fiber_graph.add_edges_from((u, v, attrs) for u, v, attrs in graph.edges(data=True)
                               if attrs["link_type"] == "fiber")
    mumbai_kolkata_path = nx.shortest_path(fiber_graph, "Mumbai", "Kolkata",
                                           weight="distance_km")
    folium.PolyLine(
        [graph.nodes[node]["pos"] for node in mumbai_kolkata_path],
        color="#8e5bb7", weight=4, opacity=0.85,
        tooltip="Modeled fiber connection: Mumbai → Jaipur → Delhi → Kolkata (no direct corridor)",
    ).add_to(connectivity_layer)

    route_points = [graph.nodes[node]["pos"] for node in route]
    folium.PolyLine(
        route_points,
        color="#d1493f",
        weight=5,
        opacity=0.95,
        tooltip=f"Epoch 50 GNN route | {len(route) - 1} hops",
    ).add_to(route_layer)
    for i, node in enumerate(route):
        if node in CITIES:
            continue
        lat, lon = graph.nodes[node]["pos"]
        folium.CircleMarker(
            [lat, lon], radius=2.6, color="#a42d26", weight=1,
            fill=True, fill_color="#fff7f0", fill_opacity=0.95,
            tooltip=f"Route relay {i}: {node}",
        ).add_to(route_layer)

    fiber_layer.add_to(m)
    fso_layer.add_to(m)
    connectivity_layer.add_to(m)
    route_layer.add_to(m)

    for city in CITIES:
        lat, lon = CITY_COORDS[city]
        is_endpoint = city in (route[0], route[-1])
        color = "#d1493f" if is_endpoint else "#163a5f"
        folium.CircleMarker(
            [lat, lon], radius=6 if is_endpoint else 5, color="white", weight=1.5,
            fill=True, fill_color=color, fill_opacity=1,
            tooltip=f"{city} (approximate city-center coordinate)",
        ).add_to(m)
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(html=(
                '<div style="font:600 12px Arial,sans-serif;color:#172b3a;'
                'text-shadow:0 1px 2px white,1px 0 2px white,-1px 0 2px white;'
                'white-space:nowrap;transform:translate(8px,-7px);">'
                f"{city}</div>")),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    title = "QKD routing experiment - epoch 50"
    note = (
        "Offline geographic schematic: no basemap tiles or API key. City coordinates "
        "are approximate and relays are linearly interpolated. Mumbai connects to "
        "Kolkata through Jaipur and Delhi; no direct corridor is modeled."
    )
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:12px;left:12px;z-index:9999;background:#fff;'
        f'padding:9px 13px;border:1px solid #9aa7b1;border-radius:4px;max-width:580px;'
        f'box-shadow:0 1px 4px #777;font-family:Arial,sans-serif;">'
        f'<div style="font-size:16px;font-weight:700">{title}</div>'
        f'<div style="font-size:11px;color:#444;margin-top:3px">{note}</div></div>'
    ))
    m.get_root().header.add_child(folium.Element(
        '<style>.leaflet-container{background:#f4f7f8!important;}</style>'
    ))
    m.fit_bounds([[8.0, 68.0], [31.5, 92.5]])
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Experiment run directory")
    parser.add_argument("--output", type=Path, help="Output HTML path (defaults to <run-dir>/plots/geographic_route_map.html)")
    args = parser.parse_args()
    output = args.output or args.run_dir / "plots" / "geographic_route_map.html"
    print(create_map(args.run_dir, output))


if __name__ == "__main__":
    main()
