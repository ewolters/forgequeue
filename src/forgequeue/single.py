"""Single-server queues — M/M/1, M/D/1, M/G/1."""

from __future__ import annotations

from dataclasses import dataclass

from forgerender import ResultMixin


@dataclass
class QueueMetrics(ResultMixin):
    """Standard queueing metrics."""

    utilization: float = 0.0  # ρ = λ / (c·μ)
    p0: float = 0.0  # P(system empty)
    lq: float = 0.0  # avg customers in queue
    ls: float = 0.0  # avg customers in system (L)
    wq: float = 0.0  # avg wait time in queue
    w: float = 0.0  # avg time in system
    stable: bool = True

    @property
    def summary(self) -> str:
        if not self.stable:
            return "Queue unstable (rho >= 1)"
        return f"Queue: rho={self.utilization:.2f} L={self.ls:.2f} W={self.w:.2f}"

    def flow(self) -> dict:
        """Flow-dialect view (forgerender.FLOW): time in system is the cycle
        time (bridge token), plus server utilization."""
        return {"cycle_time": self.w, "utilization": self.utilization}

    def to_render(self):
        """Theme-neutral ChartSpec: customers in queue vs in system."""
        from forgerender import ROLE_DATA, ChartSpec

        spec = ChartSpec(
            title="Queue Metrics",
            chart_type="queue_metrics",
            x_axis={"label": "", "grid": False},
            y_axis={"label": "Customers", "grid": True},
        )
        spec.add_trace(
            ["In Queue (Lq)", "In System (L)"], [self.lq, self.ls],
            name="Customers", trace_type="bar", color="", role=ROLE_DATA,
        )
        return spec


def mm1(arrival_rate: float, service_rate: float) -> QueueMetrics:
    """M/M/1 queue — single server, Poisson arrivals, exponential service."""
    if arrival_rate <= 0 or service_rate <= 0:
        return QueueMetrics(stable=False)

    rho = arrival_rate / service_rate
    if rho >= 1:
        return QueueMetrics(utilization=rho, stable=False)

    p0 = 1 - rho
    lq = rho ** 2 / (1 - rho)
    ls = rho / (1 - rho)
    wq = lq / arrival_rate
    w = ls / arrival_rate

    return QueueMetrics(utilization=rho, p0=p0, lq=lq, ls=ls, wq=wq, w=w)


def md1(arrival_rate: float, service_rate: float) -> QueueMetrics:
    """M/D/1 queue — deterministic service. Lq = ρ²/(2(1-ρ))."""
    if arrival_rate <= 0 or service_rate <= 0:
        return QueueMetrics(stable=False)

    rho = arrival_rate / service_rate
    if rho >= 1:
        return QueueMetrics(utilization=rho, stable=False)

    lq = rho ** 2 / (2 * (1 - rho))
    ls = lq + rho
    wq = lq / arrival_rate
    w = ls / arrival_rate
    p0 = 1 - rho

    return QueueMetrics(utilization=rho, p0=p0, lq=lq, ls=ls, wq=wq, w=w)


def mg1(arrival_rate: float, service_rate: float, cv_service: float = 1.0) -> QueueMetrics:
    """M/G/1 queue — Pollaczek-Khinchine. Lq = ρ²(1+Cs²)/(2(1-ρ))."""
    if arrival_rate <= 0 or service_rate <= 0:
        return QueueMetrics(stable=False)

    rho = arrival_rate / service_rate
    if rho >= 1:
        return QueueMetrics(utilization=rho, stable=False)

    lq = rho ** 2 * (1 + cv_service ** 2) / (2 * (1 - rho))
    ls = lq + rho
    wq = lq / arrival_rate
    w = ls / arrival_rate
    p0 = 1 - rho

    return QueueMetrics(utilization=rho, p0=p0, lq=lq, ls=ls, wq=wq, w=w)
