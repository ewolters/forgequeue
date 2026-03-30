"""Staffing optimization — optimal servers given cost/SLA targets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .multi import mmc, erlang_c


@dataclass
class StaffingRow:
    """One row in a staffing table."""

    servers: int
    utilization: float
    p_wait: float
    avg_wait: float
    avg_system_time: float
    avg_queue_length: float
    cost: float | None = None


@dataclass
class StaffingResult:
    """Optimal staffing result."""

    optimal_servers: int
    table: list[StaffingRow] = field(default_factory=list)
    min_servers: int = 0  # minimum for stability
    target_met: bool = False


def optimal_servers(
    arrival_rate: float,
    service_rate: float,
    target_wait: float = 0.0,
    target_prob: float = 0.8,
) -> int:
    """Find minimum servers to meet service level target.

    Target: P(wait ≤ target_wait) ≥ target_prob.

    Args:
        arrival_rate: λ.
        service_rate: μ per server.
        target_wait: Maximum acceptable wait time (0 = immediate service).
        target_prob: Probability of meeting the wait target (e.g., 0.80 = 80%).

    Returns:
        Minimum number of servers.
    """
    traffic = arrival_rate / service_rate
    min_c = max(1, math.ceil(traffic))  # need at least ceil(A) for stability

    for c in range(min_c, min_c + 100):
        m = mmc(arrival_rate, service_rate, c)
        if not m.stable:
            continue

        if target_wait == 0:
            # P(immediate service) = 1 - Erlang_C
            p_service = 1 - erlang_c(traffic, c)
        else:
            # P(wait ≤ t) = 1 - Erlang_C × exp(-c·μ·(1-ρ)·t)
            ec = erlang_c(traffic, c)
            rho = traffic / c
            p_service = 1 - ec * math.exp(-c * service_rate * (1 - rho) * target_wait)

        if p_service >= target_prob:
            return c

    return min_c + 100


def staffing_table(
    arrival_rate: float,
    service_rate: float,
    min_servers: int | None = None,
    max_servers: int | None = None,
    wage_rate: float | None = None,
) -> list[StaffingRow]:
    """Generate staffing table showing metrics for each server count.

    Args:
        arrival_rate: λ.
        service_rate: μ per server.
        min_servers: Start of range (default: ceil(traffic)).
        max_servers: End of range (default: min + 10).
        wage_rate: Cost per server per time unit (for cost column).

    Returns:
        List of StaffingRow, one per server count.
    """
    traffic = arrival_rate / service_rate
    if min_servers is None:
        min_servers = max(1, math.ceil(traffic))
    if max_servers is None:
        max_servers = min_servers + 10

    rows = []
    for c in range(min_servers, max_servers + 1):
        m = mmc(arrival_rate, service_rate, c)
        if not m.stable:
            continue

        ec = erlang_c(traffic, c)
        cost = wage_rate * c if wage_rate else None

        rows.append(StaffingRow(
            servers=c,
            utilization=m.utilization,
            p_wait=ec,
            avg_wait=m.wq,
            avg_system_time=m.w,
            avg_queue_length=m.lq,
            cost=cost,
        ))

    return rows


def staffing_cost(
    servers: int,
    wage_rate: float,
    arrival_rate: float,
    service_rate: float,
    wait_cost: float = 0.0,
) -> dict:
    """Total cost analysis for a given staffing level.

    Total cost = server cost + waiting cost.

    Args:
        servers: Number of servers.
        wage_rate: Cost per server per time unit.
        arrival_rate: λ.
        service_rate: μ.
        wait_cost: Cost per customer per time unit waiting.

    Returns:
        Dict with server_cost, waiting_cost, total_cost.
    """
    m = mmc(arrival_rate, service_rate, servers)
    server_cost = servers * wage_rate
    waiting_cost_total = arrival_rate * m.wq * wait_cost if m.stable else float("inf")
    total = server_cost + waiting_cost_total

    return {
        "servers": servers,
        "server_cost": server_cost,
        "waiting_cost": waiting_cost_total,
        "total_cost": total,
        "utilization": m.utilization if m.stable else 0,
        "avg_wait": m.wq if m.stable else float("inf"),
    }
