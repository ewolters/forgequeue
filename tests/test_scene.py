"""forgequeue consumes the shared forgecore.Scene: nodes -> queue nodes,
edges' routing_prob -> the Jackson routing matrix. jackson_network untouched."""

from forgecore import Edge, Node, Scene

from forgequeue.network import jackson_network
from forgequeue.scene import network_from_scene


def _scene() -> Scene:
    # A receives external arrivals; half of A's output routes to B.
    return Scene(
        nodes=[
            Node(id="A", kind="station",
                 attrs={"servers": 1, "service_rate": 10.0, "arrival_rate": 5.0}),
            Node(id="B", kind="station",
                 attrs={"servers": 1, "service_rate": 10.0}),
        ],
        edges=[Edge(from_id="A", to_id="B", attrs={"routing_prob": 0.5})],
    )


def test_adapter_matches_a_direct_jackson_call():
    # The adapter must only wire inputs — identical result to calling the
    # solver by hand with the same nodes / routing matrix / external arrivals.
    direct = jackson_network(
        nodes=[{"name": "A", "servers": 1, "service_rate": 10.0},
               {"name": "B", "servers": 1, "service_rate": 10.0}],
        routing_matrix=[[0.0, 0.5], [0.0, 0.0]],
        external_arrivals=[5.0, 0.0],
    )
    adapted = network_from_scene(_scene())
    assert adapted.stable == direct.stable
    assert [s.metrics.utilization for s in adapted.stages] == \
           [s.metrics.utilization for s in direct.stages]


def test_traffic_equations_give_expected_utilization():
    r = network_from_scene(_scene())
    # lambda_A = 5 ; lambda_B = 0.5*5 = 2.5 ; rho = lambda / (c*mu)
    assert r.stable is True
    assert round(r.stages[0].metrics.utilization, 4) == 0.5
    assert round(r.stages[1].metrics.utilization, 4) == 0.25


def test_stage_names_are_node_ids_so_result_keys_back():
    r = network_from_scene(_scene())
    assert [s.name for s in r.stages] == ["A", "B"]


def test_missing_edges_are_zero_probability():
    scene = Scene(
        nodes=[Node(id="X", attrs={"service_rate": 10.0, "arrival_rate": 3.0}),
               Node(id="Y", attrs={"service_rate": 10.0})],
        edges=[],  # no routing -> Y gets no traffic
    )
    r = network_from_scene(scene)
    assert round(r.stages[1].metrics.utilization, 4) == 0.0
