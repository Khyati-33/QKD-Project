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
from folium.plugins import Fullscreen

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
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )
    Fullscreen(position="topright", title="Expand map", title_cancel="Exit full screen").add_to(m)

    fiber_layer = folium.FeatureGroup(name="Modeled fiber backbone (80 km/link)", show=True)
    fso_layer = folium.FeatureGroup(name="Modeled FSO detours (10 km/link)", show=True)
    route_layer = folium.FeatureGroup(name="Epoch 50 learned route", show=True)

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
    route_layer.add_to(m)

    for city in CITIES:
        lat, lon = CITY_COORDS[city]
        is_endpoint = city in (route[0], route[-1])
        folium.Marker(
            [lat, lon],
            tooltip=city,
            popup=folium.Popup(f"<b>{city}</b><br>Approximate city-center coordinate", max_width=240),
            icon=folium.Icon(color="red" if is_endpoint else "darkblue", icon="info-sign"),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    title = "QKD routing experiment · epoch 50"
    note = (
        "Illustrative map of a synthetic topology. City coordinates are approximate; "
        "relay positions are linearly interpolated. Corridor geometry and modeled "
        "link lengths are not surveyed infrastructure."
    )
    m.get_root().html.add_child(folium.Element(
        f'''<div style="position:fixed;top:12px;left:50px;z-index:9999;background:#fff;"
        "padding:9px 13px;border:1px solid #aaa;border-radius:4px;max-width:590px;"
        "box-shadow:0 1px 4px #777;font-family:Arial,sans-serif;">
        <div style="font-size:16px;font-weight:700">{title}</div>
        <div style="font-size:11px;color:#444;margin-top:3px">{note}</div></div>'''
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
