import networkx as nx

from topology import (CITIES, CORRIDORS, CORRIDOR_PAIRS, DEFAULT_DESTINATION,
                      DEFAULT_SOURCE, FIBER_SPAN_KM, FSO_SPAN_KM, HUB_CITIES,
                      build_topology, shortest_fiber_route, topology_summary)


def test_topology_targets():
    graph = build_topology()
    summary = topology_summary(graph)
    assert set(CITIES) <= set(graph.nodes)
    assert nx.is_connected(graph)
    assert summary["nodes"] > len(CITIES)
    assert summary["links"] == summary["fiber_links"] + summary["fso_links"]
    assert summary["city_corridors"] == len(CORRIDORS) == 17
    assert 23 <= summary["default_route_hops"] <= 35
    assert 2000 <= summary["default_route_km"] <= 2300
    assert any(d["node_type"] == "full_tn" for _, d in graph.nodes(data=True))
    assert any(d["node_type"] == "stn" for _, d in graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        if data["backbone"]:
            assert data["link_type"] == "fiber"
            assert 0 < data["distance_km"] <= FIBER_SPAN_KM
        else:
            assert data["link_type"] == "fso"
            assert 0 < data["distance_km"] <= FSO_SPAN_KM
            assert data["visibility_assumption"] == "unverified_clear_line_of_sight"
    assert len(shortest_fiber_route(graph, DEFAULT_SOURCE, DEFAULT_DESTINATION)) - 1 <= 35
    for hub in HUB_CITIES:
        for city in CITIES:
            if city != hub:
                pair = tuple(sorted((hub, city)))
                assert pair in CORRIDOR_PAIRS
                assert nx.has_path(graph, hub, city)
    assert tuple(sorted(("Mumbai", "Hyderabad"))) in CORRIDOR_PAIRS
    assert tuple(sorted(("Mumbai", "Kolkata"))) in CORRIDOR_PAIRS
