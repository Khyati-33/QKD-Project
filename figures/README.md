# Road-aligned topology map

`india_road_corridor_map.html` shows the revised network: 17 road-routed city
corridors, fiber relays at no more than 80 km along each route, and optional
candidate FSO paths with links no longer than 10 km. Jaipur, Hyderabad, and
Kolkata each connect by a routed corridor to every other modeled city. Mumbai
has direct candidate corridors to Hyderabad and Kolkata.

The red path is the **shortest fiber baseline**, not a learned-policy result.
The previous `epoch50_geographic_route_map.html` is an archived view of the
older abstract topology; its 50-epoch route metrics do not apply to this revised
graph. The revised network needs a new training and evaluation run.

Suggested paper caption: *Proposed QKD backbone over fastest OpenStreetMap road
corridors among seven Indian metropolitan nodes. Fiber repeater spacing is at
most 80 km; candidate FSO links are at most 10 km and assume unverified line of
sight. The displayed paths are road-alignment proxies, not verified telecom
fiber routes. Map data from OpenStreetMap contributors,
<https://www.openstreetmap.org/copyright>.*

The HTML map does not request basemap tiles or a tile API key. Regenerate it
from the committed route snapshot with:

```powershell
python make_folium_map.py
```

For source, route retrieval, and licensing details, see [data/README.md](../data/README.md).
