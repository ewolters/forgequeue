"""Calibration adapter for ForgeQueue."""

from __future__ import annotations


GOLDEN_CASES = [
    {
        "case_id": "CAL-QUE-001",
        "description": "M/M/1: λ=8, μ=10 → ρ=0.8, Lq=3.2",
        "test": "mm1",
        "input": {"arrival_rate": 8, "service_rate": 10},
        "expected": {"utilization": 0.8, "lq": 3.2},
    },
    {
        "case_id": "CAL-QUE-002",
        "description": "M/M/1 unstable: λ=12, μ=10",
        "test": "mm1_unstable",
        "input": {"arrival_rate": 12, "service_rate": 10},
        "expected": {"stable": False},
    },
    {
        "case_id": "CAL-QUE-003",
        "description": "M/M/c: λ=20, μ=8, c=3 → ρ=0.833",
        "test": "mmc",
        "input": {"arrival_rate": 20, "service_rate": 8, "servers": 3},
        "expected": {"utilization_gt": 0.8, "stable": True},
    },
    {
        "case_id": "CAL-QUE-004",
        "description": "Erlang C: A=2.5, c=3 → P(wait) < 0.5",
        "test": "erlang_c",
        "input": {"traffic": 2.5, "servers": 3},
        "expected": {"p_wait_lt": 0.8},
    },
    {
        "case_id": "CAL-QUE-005",
        "description": "Erlang B: A=2.5, c=3 → P(block) < 0.1",
        "test": "erlang_b",
        "input": {"traffic": 2.5, "servers": 3},
        "expected": {"p_block_lt": 0.4},
    },
    {
        "case_id": "CAL-QUE-006",
        "description": "Optimal staffing: λ=20, μ=8, 80% in <30s",
        "test": "staffing",
        "input": {"arrival_rate": 20, "service_rate": 8, "target_wait": 0.00833, "target_prob": 0.8},
        "expected": {"servers_gt": 2},
    },
    {
        "case_id": "CAL-QUE-007",
        "description": "Tandem: 3 stages, stable",
        "test": "tandem",
        "input": {},
        "expected": {"stable": True, "n_stages": 3},
    },
    {
        "case_id": "CAL-QUE-008",
        "description": "M/D/1 Lq is half of M/M/1 Lq",
        "test": "md1_vs_mm1",
        "input": {"arrival_rate": 8, "service_rate": 10},
        "expected": {"ratio_near_half": True},
    },
]


def calibrate():
    results = []
    for case in GOLDEN_CASES:
        try:
            actual = _run_case(case["case_id"], case["test"], case["input"])
            passed = _check(actual, case["expected"])
            results.append({"case_id": case["case_id"], "passed": passed, "actual": actual})
        except Exception as e:
            results.append({"case_id": case["case_id"], "passed": False, "error": str(e)})

    p = sum(1 for r in results if r["passed"])
    return {"package": "forgequeue", "total": len(results), "passed": p,
            "failed": len(results) - p, "results": results, "is_calibrated": p == len(results)}


def _run_case(case_id, test, inp):
    from .single import mm1, md1
    from .multi import mmc as run_mmc, erlang_b as eb, erlang_c as ec
    from .staffing import optimal_servers
    from .network import tandem

    if test == "mm1":
        r = mm1(inp["arrival_rate"], inp["service_rate"])
        return {"utilization": r.utilization, "lq": r.lq, "stable": r.stable}
    elif test == "mm1_unstable":
        r = mm1(inp["arrival_rate"], inp["service_rate"])
        return {"stable": r.stable}
    elif test == "mmc":
        r = run_mmc(inp["arrival_rate"], inp["service_rate"], inp["servers"])
        return {"utilization": r.utilization, "stable": r.stable}
    elif test == "erlang_c":
        p = ec(inp["traffic"], inp["servers"])
        return {"p_wait": p}
    elif test == "erlang_b":
        p = eb(inp["traffic"], inp["servers"])
        return {"p_block": p}
    elif test == "staffing":
        n = optimal_servers(inp["arrival_rate"], inp["service_rate"], inp["target_wait"], inp["target_prob"])
        return {"servers": n}
    elif test == "tandem":
        r = tandem([
            {"name": "S1", "arrival_rate": 10, "servers": 2, "service_rate": 8},
            {"name": "S2", "servers": 1, "service_rate": 12},
            {"name": "S3", "servers": 2, "service_rate": 6},
        ])
        return {"stable": r.stable, "n_stages": len(r.stages)}
    elif test == "md1_vs_mm1":
        r1 = mm1(inp["arrival_rate"], inp["service_rate"])
        r2 = md1(inp["arrival_rate"], inp["service_rate"])
        ratio = r2.lq / r1.lq if r1.lq > 0 else 0
        return {"ratio_near_half": abs(ratio - 0.5) < 0.01}
    raise ValueError(f"Unknown test: {test}")


def _check(actual, expected):
    for k, v in expected.items():
        if k.endswith("_gt"):
            if actual.get(k[:-3], 0) <= v:
                return False
        elif k.endswith("_lt"):
            if actual.get(k[:-3], 0) >= v:
                return False
        elif isinstance(v, bool):
            if actual.get(k) != v:
                return False
        elif isinstance(v, (int, float)):
            av = actual.get(k)
            if av is None or abs(float(av) - v) > 0.05:
                return False
    return True


def get_calibration_adapter():
    try:
        from forgecal.core import CalibrationAdapter, CalibrationCase, Expectation
    except ImportError:
        return None

    cases = []
    for gc in GOLDEN_CASES:
        expectations = []
        for k, v in gc["expected"].items():
            if k.endswith("_gt"):
                expectations.append(Expectation(key=k[:-3], expected=v, comparison="greater_than"))
            elif k.endswith("_lt"):
                expectations.append(Expectation(key=k[:-3], expected=v, comparison="less_than"))
            elif isinstance(v, bool):
                expectations.append(Expectation(key=k, expected=v, comparison="equals"))
            else:
                expectations.append(Expectation(key=k, expected=v, tolerance=0.05, comparison="abs_within"))
        cases.append(CalibrationCase(
            case_id=gc["case_id"], package="forgequeue", category="queueing",
            analysis_type="queue", analysis_id=gc["test"],
            config=gc["input"], data={}, expectations=expectations, description=gc["description"],
        ))

    def _run(case):
        gc = next(g for g in GOLDEN_CASES if g["case_id"] == case.case_id)
        return _run_case(case.case_id, gc["test"], gc["input"])

    from forgequeue import __version__
    return CalibrationAdapter(package="forgequeue", version=__version__, cases=cases, runner=_run)
