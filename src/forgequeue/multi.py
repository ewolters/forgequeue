"""Multi-server queues — M/M/c, M/M/c/K, Erlang B/C."""

from __future__ import annotations

import math

from .single import QueueMetrics


def _factorial(n: int) -> float:
    return float(math.factorial(n))


def mmc(arrival_rate: float, service_rate: float, servers: int) -> QueueMetrics:
    """M/M/c queue — multiple identical servers.

    Args:
        arrival_rate: λ.
        service_rate: μ per server.
        servers: c (number of servers).
    """
    c = servers
    lam = arrival_rate
    mu = service_rate
    r = lam / mu  # traffic intensity
    rho = r / c  # per-server utilization

    if rho >= 1:
        return QueueMetrics(utilization=rho, stable=False)

    # P₀ — probability system is empty
    sum_terms = sum(r ** n / _factorial(n) for n in range(c))
    last_term = (r ** c / _factorial(c)) * (1 / (1 - rho))
    p0 = 1 / (sum_terms + last_term)

    # P(wait) — Erlang C
    pc = (r ** c / _factorial(c)) * (1 / (1 - rho)) * p0

    # Metrics
    lq = pc * rho / (1 - rho)
    wq = lq / lam if lam > 0 else 0.0
    w = wq + 1 / mu
    ls = lam * w

    return QueueMetrics(utilization=rho, p0=p0, lq=lq, ls=ls, wq=wq, w=w)


def mmck(
    arrival_rate: float,
    service_rate: float,
    servers: int,
    capacity: int,
) -> QueueMetrics:
    """M/M/c/K queue — finite buffer (capacity K total in system).

    Customers turned away when system has K customers.

    Args:
        arrival_rate: λ.
        service_rate: μ per server.
        servers: c.
        capacity: K (max customers in system, including being served).
    """
    c = servers
    K = capacity
    lam = arrival_rate
    mu = service_rate
    r = lam / mu

    if c <= 0 or K < c:
        return QueueMetrics(stable=False)

    rho = r / c

    # P₀
    sum1 = sum(r ** n / _factorial(n) for n in range(c))

    if abs(rho - 1.0) < 1e-10:
        sum2 = (r ** c / _factorial(c)) * (K - c + 1)
    else:
        sum2 = (r ** c / _factorial(c)) * (1 - rho ** (K - c + 1)) / (1 - rho)

    p0 = 1 / (sum1 + sum2)

    # P(K) — probability of full system (blocking)
    if K < c:
        pK = (r ** K / _factorial(K)) * p0
    else:
        pK = (r ** K / (_factorial(c) * c ** (K - c))) * p0

    # Effective arrival rate (some customers blocked)
    lam_eff = lam * (1 - pK)

    # L — average customers in system
    ls_total = 0.0
    for n in range(K + 1):
        if n < c:
            pn = (r ** n / _factorial(n)) * p0
        else:
            pn = (r ** n / (_factorial(c) * c ** (n - c))) * p0
        ls_total += n * pn

    w = ls_total / lam_eff if lam_eff > 0 else 0
    ls_serving = min(ls_total, c * rho)
    lq = max(0, ls_total - ls_serving)
    wq = lq / lam_eff if lam_eff > 0 else 0

    return QueueMetrics(
        utilization=rho if rho < 1 else float(lam_eff / (c * mu)),
        p0=p0, lq=lq, ls=ls_total, wq=wq, w=w,
        stable=True,  # finite buffer is always stable
    )


def erlang_b(traffic: float, servers: int) -> float:
    """Erlang B formula — blocking probability (no waiting room).

    P(block) for M/M/c/c system (loss system).

    Args:
        traffic: A = λ/μ (offered traffic in Erlangs).
        servers: c (number of servers/circuits).

    Returns:
        Blocking probability.
    """
    # Recursive computation (numerically stable)
    b = 1.0
    for n in range(1, servers + 1):
        b = (traffic * b) / (n + traffic * b)
    return b


def erlang_c(traffic: float, servers: int) -> float:
    """Erlang C formula — probability of waiting.

    P(wait > 0) for M/M/c system.

    Args:
        traffic: A = λ/μ (offered traffic in Erlangs).
        servers: c.

    Returns:
        Probability a customer must wait. Returns 1.0 if unstable (A ≥ c).
    """
    if traffic >= servers:
        return 1.0

    c = servers
    A = traffic

    num = (A ** c / _factorial(c)) * (c / (c - A))
    denom = sum(A ** k / _factorial(k) for k in range(c)) + num
    return num / denom
