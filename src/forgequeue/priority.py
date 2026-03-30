"""Priority queues — non-preemptive multi-class."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PriorityClassMetrics:
    """Metrics for one priority class."""

    priority: int
    arrival_rate: float
    service_rate: float
    wq: float = 0.0  # avg wait in queue
    w: float = 0.0  # avg time in system
    lq: float = 0.0  # avg in queue


@dataclass
class PriorityQueueResult:
    """Priority queue result — per-class and system metrics."""

    classes: list[PriorityClassMetrics] = field(default_factory=list)
    total_utilization: float = 0.0
    stable: bool = True


def priority_queue(
    classes: list[dict],
) -> PriorityQueueResult:
    """Non-preemptive priority queue (M/M/c with priority classes).

    Higher priority (lower number) gets served first from queue.
    Uses accumulating priority formula.

    Args:
        classes: List of {priority: int, arrival_rate: float, service_rate: float}.
            Priority 1 = highest, 2 = next, etc.

    Returns:
        PriorityQueueResult with per-class metrics.
    """
    sorted_classes = sorted(classes, key=lambda c: c["priority"])

    # Total utilization
    rho_total = sum(c["arrival_rate"] / c["service_rate"] for c in sorted_classes)
    if rho_total >= 1:
        return PriorityQueueResult(total_utilization=rho_total, stable=False)

    # W₀ = mean unfinished work = Σ ρᵢ / (2μᵢ)
    # Actually: W₀ = Σ λᵢ E[S²ᵢ] / 2 = Σ λᵢ / μᵢ² (for exponential service)
    w0 = sum(c["arrival_rate"] / c["service_rate"] ** 2 for c in sorted_classes)

    results = []
    sigma_prev = 0.0  # cumulative utilization of higher priority classes

    for i, cls in enumerate(sorted_classes):
        lam = cls["arrival_rate"]
        mu = cls["service_rate"]
        rho_i = lam / mu

        sigma_curr = sigma_prev + rho_i

        # Wq for class i (non-preemptive)
        denom = (1 - sigma_prev) * (1 - sigma_curr)
        wq = w0 / denom if denom > 0 else float("inf")
        w = wq + 1 / mu
        lq = lam * wq

        results.append(PriorityClassMetrics(
            priority=cls["priority"],
            arrival_rate=lam,
            service_rate=mu,
            wq=wq,
            w=w,
            lq=lq,
        ))

        sigma_prev = sigma_curr

    return PriorityQueueResult(
        classes=results,
        total_utilization=rho_total,
        stable=True,
    )
