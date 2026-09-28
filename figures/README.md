# Epoch 50 route map

`epoch50_geographic_route_map.html` is an interactive Folium map generated from
the completed 50-epoch run. It shows the modeled fiber backbone, modeled FSO
detours, the final GNN route, and the modeled Mumbai–Kolkata connection via
Jaipur and Delhi in separate toggleable layers. It does not request basemap
tiles or an API key.

Suggested paper caption: *Geographic visualization of the epoch 50 GNN route
for the Delhi–Chennai evaluation in the synthetic QKD topology. City positions
are approximate and relay positions are linearly interpolated; link geometry
does not represent surveyed infrastructure.*

The map uses an offline latitude/longitude graticule. Its corridors are
synthetic and are not surveyed infrastructure. Regenerate it with:

```powershell
python make_folium_map.py --run-dir experiments/runs/defence_monsoon_night_50ep_idq_20260928T044715Z_c84be8c5 --output figures/epoch50_geographic_route_map.html
```
