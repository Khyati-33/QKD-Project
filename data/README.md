# Routed road corridor data

`osm_road_corridors.json` contains 17 routed city-pair geometries. The set
includes direct routes from each of Jaipur, Hyderabad, and Kolkata to every
other modeled city, including Mumbai–Hyderabad and Mumbai–Kolkata, plus the
Mumbai–Bangalore and Bangalore–Chennai corridors retained from the original
backbone.

The geometry was fetched on 2026-09-28 from the public OSRM routing demo, which
routes over OpenStreetMap road data. Each route is the fastest driving route
between approximate city-center coordinates. Routes use a mixture of roads;
they are not filtered to National Highways alone. The stored road references
include National and State Highway labels where OSM has them. Coordinates were
simplified to a 60 m maximum-deviation tolerance. Source geometry, route
distances, and road-name/reference summaries are recorded in the JSON.

The graph places a fiber relay at intervals no greater than 80 km along each
routed road polyline. Candidate FSO relay paths follow the same corridors at
approximately 9.9 km spacing; every modeled straight-line FSO link is checked
to remain at or below 10 km. This spacing margin avoids rounding/simplification
overshoot. The FSO paths assume clear line of sight: no terrain, vegetation,
building, or site survey was used to validate visibility. City-center access
connectors are approximate.

The Department of Telecommunications reports that public-sector, state, and
private optical-fiber assets have been mapped on the PM GatiShakti National
Master Plan platform, but detailed corridor centerlines were not available in
the public sources used for this project. The public NATMO national-highway
dataset metadata describes a 1:14,000,000 map derived from 2014 mapping and
digitized in 2019, which is too coarse for relay placement. This project
therefore uses an explicitly labeled road-alignment proxy, not a claim about
actual operator cable routes.

## Attribution and data license

Road geometry is derived from OpenStreetMap. Credit it as “Map data from
OpenStreetMap contributors” and link to
<https://www.openstreetmap.org/copyright> in publications. The OSM-derived
route dataset is provided under the [Open Database License 1.0
(ODbL)](https://opendatacommons.org/licenses/odbl/1-0/); consult the license
for attribution and share-alike obligations. The OSRM route service is
documented at <https://project-osrm.org/docs/v5.24.0/api/>.

To refresh the routes from the public demo, run
`python scripts/fetch_osm_corridors.py`. This contacts the external OSRM demo
service, waits between requests, and records the retrieval timestamp. The
stored snapshot makes graph construction reproducible without a network call.
