"""forgequeue conforms to the FLOW dialect — the queueing solver speaks the
full-rate tokens (utilization) plus the cycle_time bridge, and renders."""

from forgerender import (
    FLOW,
    ROLE_OUT_OF_CONTROL,
    ChartSpec,
    Result,
    result_registry,
    speaks,
)

from forgequeue.network import NetworkResult, tandem
from forgequeue.single import QueueMetrics, mm1


def _net():
    return tandem([
        {"name": "A", "servers": 1, "service_rate": 3.0, "arrival_rate": 2.0},
        {"name": "B", "servers": 1, "service_rate": 4.0},
    ])


def test_queue_metrics_conforms_to_result_protocol():
    assert isinstance(mm1(2.0, 3.0), Result)


def test_queue_flow_view_speaks_the_flow_dialect():
    assert speaks(mm1(2.0, 3.0).flow(), FLOW)


def test_queue_flow_bridges_cycle_time_to_time_in_system():
    m = mm1(2.0, 3.0)
    assert m.flow()["cycle_time"] == m.w


def test_queue_to_render_is_a_chartspec():
    assert isinstance(mm1(2.0, 3.0).to_render(), ChartSpec)


def test_network_result_conforms_to_result_protocol():
    assert isinstance(_net(), Result)


def test_network_flow_view_speaks_the_flow_dialect():
    assert speaks(_net().flow(), FLOW)


def test_network_to_render_flags_the_bottleneck():
    spec = _net().to_render()
    assert isinstance(spec, ChartSpec)
    assert any(m.role == ROLE_OUT_OF_CONTROL for m in spec.markers)


def test_importing_forgequeue_registers_its_result_types():
    reg = result_registry()
    assert reg.get("QueueMetrics") is QueueMetrics
    assert reg.get("NetworkResult") is NetworkResult
