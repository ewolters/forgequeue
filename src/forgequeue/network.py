"""Network queues — tandem (serial) and Jackson networks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .single import QueueMetrics
from .multi import mmc


@dataclass
class StageMetrics:
    """Metrics for one stage in a tandem/network queue."""

    name: str
    metrics: QueueMetrics
    arrival_rate: float = 0.0


@dataclass
class NetworkResult:
    """Queueing network result."""

    stages: list[StageMetrics] = field(default_factory=list)
    total_wip: float = 0.0  # total customers in system
    total_time: float = 0.0  # total time through system
    bottleneck: str = ""  # stage with highest utilization
    stable: bool = True


def tandem(stages: list[dict]) -> NetworkResult:
    """Tandem (serial) queue network.

    Each stage is an independent M/M/c queue. Output of stage i feeds stage i+1.
    Arrival rate is the same for all stages (by conservation of flow).

    Args:
        stages: [{name, servers, service_rate}]. First stage gets external arrivals.
            Include arrival_rate in first stage dict.

    Returns:
        NetworkResult with per-stage metrics and totals.
    """
    if not stages:
        return NetworkResult()

    arrival_rate = stages[0].get("arrival_rate", 0)
    if arrival_rate <= 0:
        return NetworkResult(stable=False)

    results = []
    total_wip = 0.0
    total_time = 0.0
    max_util = 0.0
    bottleneck_name = ""

    for stage in stages:
        name = stage.get("name", "Stage")
        servers = stage.get("servers", 1)
        mu = stage.get("service_rate", 1)

        m = mmc(arrival_rate, mu, servers)
        if not m.stable:
            return NetworkResult(stable=False)

        results.append(StageMetrics(name=name, metrics=m, arrival_rate=arrival_rate))
        total_wip += m.ls
        total_time += m.w

        if m.utilization > max_util:
            max_util = m.utilization
            bottleneck_name = name

    return NetworkResult(
        stages=results,
        total_wip=total_wip,
        total_time=total_time,
        bottleneck=bottleneck_name,
        stable=True,
    )


def jackson_network(
    nodes: list[dict],
    routing_matrix: list[list[float]],
    external_arrivals: list[float],
) -> NetworkResult:
    """Open Jackson network — nodes with probabilistic routing.

    Solves traffic equations: λᵢ = γᵢ + Σⱼ λⱼ pⱼᵢ
    where γᵢ = external arrival rate, pⱼᵢ = routing probability j→i.

    Each node is then an independent M/M/c queue with its solved λᵢ.

    Args:
        nodes: [{name, servers, service_rate}] for each node.
        routing_matrix: pᵢⱼ = P(customer goes from node i to node j). Rows sum ≤ 1.
        external_arrivals: γᵢ = external arrival rate to each node.

    Returns:
        NetworkResult with per-node metrics.
    """
    import numpy as np

    n = len(nodes)
    if n == 0:
        return NetworkResult()

    P = np.array(routing_matrix, dtype=float)
    gamma = np.array(external_arrivals, dtype=float)

    # Solve: λ = γ + Pᵀλ  →  (I - Pᵀ)λ = γ
    eye = np.eye(n)
    try:
        lam = np.linalg.solve(eye - P.T, gamma)
    except np.linalg.LinAlgError:
        return NetworkResult(stable=False)

    results = []
    total_wip = 0.0
    total_time = 0.0
    max_util = 0.0
    bottleneck_name = ""

    for i, node in enumerate(nodes):
        name = node.get("name", f"Node {i+1}")
        servers = node.get("servers", 1)
        mu = node.get("service_rate", 1)
        arrival = float(lam[i])

        m = mmc(arrival, mu, servers)
        if not m.stable:
            return NetworkResult(stable=False)

        results.append(StageMetrics(name=name, metrics=m, arrival_rate=arrival))
        total_wip += m.ls
        total_time += m.w

        if m.utilization > max_util:
            max_util = m.utilization
            bottleneck_name = name

    return NetworkResult(
        stages=results,
        total_wip=total_wip,
        total_time=total_time,
        bottleneck=bottleneck_name,
        stable=True,
    )
