"""Fetch approximate intercity driving corridors from the public OSRM demo API.

OSRM routes over OpenStreetMap road data. This is a road-alignment proxy for
fiber corridors, not evidence that telecom fiber exists along every segment.
The script waits between requests to respect the public demo service.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topology import CITIES, CITY_COORDS


HUBS = ("Jaipur", "Hyderabad", "Kolkata")
EXTRA_CORRIDORS = (("Mumbai", "Bangalore"), ("Bangalore", "Chennai"))
GEOMETRY_TOLERANCE_M = 60.0


def corridor_pairs() -> list[tuple[str, str]]:
    pairs = {tuple(sorted((hub, city))) for hub in HUBS for city in CITIES if city != hub}
    pairs.update(tuple(sorted(pair)) for pair in EXTRA_CORRIDORS)
    return sorted(pairs)


def simplify_geometry(coords: list[list[float]], tolerance_m: float) -> list[list[float]]:
    """Ramer-Douglas-Peucker simplification in a local metric projection."""
    if len(coords) <= 2:
        return coords
    mean_lat = sum(p[1] for p in coords) / len(coords)
    x_scale = 111_320.0 * math.cos(math.radians(mean_lat))
    y_scale = 111_320.0
    xy = [(lon * x_scale, lat * y_scale) for lon, lat in coords]
    keep = {0, len(coords) - 1}
    stack = [(0, len(coords) - 1)]
    tolerance2 = tolerance_m * tolerance_m
    while stack:
        first, last = stack.pop()
        ax, ay = xy[first]
        bx, by = xy[last]
        dx, dy = bx - ax, by - ay
        norm2 = dx * dx + dy * dy
        max_dist2 = -1.0
        split = None
        for i in range(first + 1, last):
            px, py = xy[i]
            if norm2 == 0:
                dist2 = (px - ax) ** 2 + (py - ay) ** 2
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / norm2))
                dist2 = (px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2
            if dist2 > max_dist2:
                max_dist2, split = dist2, i
        if split is not None and max_dist2 > tolerance2:
            keep.add(split)
            stack.extend(((first, split), (split, last)))
    return [coords[i] for i in sorted(keep)]


def fetch(output: Path, delay: float = 1.1) -> None:
    routes = {}
    for start, end in corridor_pairs():
        start_lat, start_lon = CITY_COORDS[start]
        end_lat, end_lon = CITY_COORDS[end]
        url = ("https://router.project-osrm.org/route/v1/driving/"
               f"{start_lon},{start_lat};{end_lon},{end_lat}")
        response = requests.get(url, params={
            "overview": "full", "geometries": "geojson", "steps": "true",
        }, headers={"User-Agent": "QKD-Project-research/1.0 (openstreetmap road-routing proxy)"},
           timeout=90)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "Ok" or not payload.get("routes"):
            raise RuntimeError(f"OSRM failed for {start} to {end}: {payload}")
        route = payload["routes"][0]
        step_details = []
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                if step.get("distance", 0) <= 0:
                    continue
                name, ref = step.get("name", ""), step.get("ref", "")
                if step_details and step_details[-1]["name"] == name and step_details[-1]["ref"] == ref:
                    step_details[-1]["distance_m"] = round(
                        step_details[-1]["distance_m"] + step["distance"], 1)
                else:
                    step_details.append({"name": name, "ref": ref,
                                         "distance_m": round(step["distance"], 1)})
        key = f"{start}--{end}"
        routes[key] = {
            "start": start,
            "end": end,
            "distance_m": round(route["distance"], 1),
            "duration_s": round(route["duration"], 1),
            "coordinates_lon_lat": [
                [round(lon, 6), round(lat, 6)]
                for lon, lat in simplify_geometry(route["geometry"]["coordinates"],
                                                   GEOMETRY_TOLERANCE_M)
            ],
            "road_steps": step_details,
        }
        print(f"{start} -> {end}: {route['distance'] / 1000:.1f} road km; "
              f"{len(route['geometry']['coordinates'])} geometry points")
        time.sleep(delay)

    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "source": "OSRM public routing demo using OpenStreetMap road data",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "attribution": "© OpenStreetMap contributors, ODbL 1.0",
        "method": "Fastest driving route between approximate city-center coordinates; "
                  "not filtered exclusively to National Highways and not a telecom OFC map.",
        "geometry_simplification_tolerance_m": GEOMETRY_TOLERANCE_M,
        "hub_cities": list(HUBS),
        "routes": routes,
    }
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/osm_road_corridors.json"))
    parser.add_argument("--delay", type=float, default=1.1)
    args = parser.parse_args()
    fetch(args.output, delay=args.delay)


if __name__ == "__main__":
    main()
