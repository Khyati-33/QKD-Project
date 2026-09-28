import networkx as nx

from topology import (CITIES, CORRIDORS, DEFAULT_DESTINATION, DEFAULT_SOURCE,
                     build_topology, shortest_fiber_route, topology_summary)


def test_topology_targets():
    graph = build_topology()
    summary = topology_summary(graph)
    assert set(CITIES) <= set(graph.nodes)
    assert nx.is_connected(graph)
    assert summary == {"nodes": 289, "links": 319, "fiber_links": 87,
                       "fso_links": 232, "default_route_hops": 23,
                       "default_route_km": 1840.0}
    assert any(d["node_type"] == "full_tn" for _, d in graph.nodes(data=True))
    assert any(d["node_type"] == "stn" for _, d in graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        if data["backbone"]:
            assert data["link_type"] == "fiber"
        else:
            assert data["link_type"] == "fso"
            assert data["distance_km"] <= 13
    assert len(shortest_fiber_route(graph, DEFAULT_SOURCE, DEFAULT_DESTINATION)) - 1 <= 25
    assert len(CORRIDORS) == 8
