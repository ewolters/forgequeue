"""Scene adapter: turn a forgecore.Scene into an open Jackson network.

Each node becomes a queue node (servers / service_rate from attrs; external
arrival_rate from attrs). Each edge's attrs['routing_prob'] becomes P[i][j] in
the routing matrix. The jackson_network solver is untouched — the Scene only
feeds it. Stage names are node ids so the result keys back to the Scene.
"""

from __future__ import annotations

from .network import NetworkResult, jackson_network


def network_from_scene(scene) -> NetworkResult:
    """Adapt a forgecore.Scene into a Jackson-network solve.

    node.attrs: servers (default 1), service_rate (default 1.0),
    arrival_rate (external, default 0.0). edge.attrs['routing_prob'] -> P[i][j]
    (default 0.0 for absent edges).
    """
    nodes = scene.nodes
    index = {n.id: i for i, n in enumerate(nodes)}
    size = len(nodes)

    node_dicts = [
        {
            "name": n.id,
            "servers": n.attrs.get("servers", 1),
            "service_rate": n.attrs.get("service_rate", 1.0),
        }
        for n in nodes
    ]

    routing_matrix = [[0.0] * size for _ in range(size)]
    for e in scene.edges:
        if e.from_id in index and e.to_id in index:
            routing_matrix[index[e.from_id]][index[e.to_id]] = e.attrs.get("routing_prob", 0.0)

    external_arrivals = [n.attrs.get("arrival_rate", 0.0) for n in nodes]

    return jackson_network(node_dicts, routing_matrix, external_arrivals)
