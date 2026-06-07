"""Tests for ForgeQueue."""

from forgequeue.single import mm1, md1, mg1
from forgequeue.multi import mmc, mmck, erlang_b, erlang_c
from forgequeue.priority import priority_queue
from forgequeue.network import tandem, jackson_network
from forgequeue.staffing import optimal_servers, staffing_table, staffing_cost


class TestMM1:
    def test_basic(self):
        r = mm1(8, 10)
        assert abs(r.utilization - 0.8) < 0.01
        assert abs(r.lq - 3.2) < 0.01
        assert abs(r.ls - 4.0) < 0.01

    def test_unstable(self):
        r = mm1(12, 10)
        assert not r.stable

    def test_light_load(self):
        r = mm1(1, 10)
        assert r.lq < 0.02


class TestMD1:
    def test_half_of_mm1(self):
        mm1_r = mm1(8, 10)
        md1_r = md1(8, 10)
        assert abs(md1_r.lq / mm1_r.lq - 0.5) < 0.01


class TestMG1:
    def test_mm1_equiv(self):
        mm1_r = mm1(8, 10)
        mg1_r = mg1(8, 10, cv_service=1.0)
        assert abs(mg1_r.lq - mm1_r.lq) < 0.01

    def test_md1_equiv(self):
        md1_r = md1(8, 10)
        mg1_r = mg1(8, 10, cv_service=0.0)
        assert abs(mg1_r.lq - md1_r.lq) < 0.01


class TestMMC:
    def test_basic(self):
        r = mmc(20, 8, 3)
        assert r.stable
        assert 0.8 < r.utilization < 0.9

    def test_single_server_matches_mm1(self):
        mm1_r = mm1(8, 10)
        mmc_r = mmc(8, 10, 1)
        assert abs(mm1_r.lq - mmc_r.lq) < 0.01

    def test_more_servers_less_wait(self):
        r2 = mmc(20, 8, 3)
        r3 = mmc(20, 8, 5)
        assert r3.wq < r2.wq

    def test_mmc_idle_queue_zero_arrivals(self):
        # lambda=0: an idle M/M/c is well-defined, not a crash (matches mmck's guard)
        r = mmc(0.0, 10.0, 1)
        assert r.stable is True
        assert r.utilization == 0.0
        assert r.wq == 0.0
        assert r.lq == 0.0


class TestMMCK:
    def test_finite_always_stable(self):
        r = mmck(100, 10, 2, 10)  # way overloaded but finite
        assert r.stable

    def test_less_wait_than_infinite(self):
        fin_r = mmck(15, 8, 2, 20)
        # Finite buffer has blocking, so effective load is lower
        assert fin_r.stable


class TestErlang:
    def test_erlang_b(self):
        pb = erlang_b(2.5, 3)
        assert 0 < pb < 0.4

    def test_erlang_c(self):
        pc = erlang_c(2.5, 3)
        assert 0 < pc < 0.8

    def test_erlang_c_unstable(self):
        pc = erlang_c(3.5, 3)
        assert pc == 1.0


class TestPriority:
    def test_higher_priority_waits_less(self):
        r = priority_queue([
            {"priority": 1, "arrival_rate": 5, "service_rate": 10},
            {"priority": 2, "arrival_rate": 3, "service_rate": 10},
        ])
        assert r.stable
        assert r.classes[0].wq < r.classes[1].wq

    def test_unstable(self):
        r = priority_queue([
            {"priority": 1, "arrival_rate": 8, "service_rate": 10},
            {"priority": 2, "arrival_rate": 5, "service_rate": 10},
        ])
        assert not r.stable


class TestTandem:
    def test_three_stages(self):
        r = tandem([
            {"name": "Cut", "arrival_rate": 10, "servers": 2, "service_rate": 8},
            {"name": "Weld", "servers": 1, "service_rate": 12},
            {"name": "Paint", "servers": 2, "service_rate": 6},
        ])
        assert r.stable
        assert len(r.stages) == 3
        assert r.bottleneck != ""

    def test_bottleneck_identified(self):
        r = tandem([
            {"name": "Fast", "arrival_rate": 5, "servers": 1, "service_rate": 10},
            {"name": "Slow", "servers": 1, "service_rate": 6},
        ])
        assert r.bottleneck == "Slow"


class TestJackson:
    def test_two_node(self):
        nodes = [
            {"name": "A", "servers": 2, "service_rate": 10},
            {"name": "B", "servers": 1, "service_rate": 15},
        ]
        routing = [[0.0, 0.5], [0.0, 0.0]]  # 50% of A goes to B
        external = [8.0, 2.0]
        r = jackson_network(nodes, routing, external)
        assert r.stable
        assert len(r.stages) == 2


class TestStaffing:
    def test_optimal(self):
        n = optimal_servers(20, 8, target_wait=0.01, target_prob=0.8)
        assert n >= 3

    def test_table(self):
        rows = staffing_table(20, 8, wage_rate=25)
        assert len(rows) > 0
        assert all(r.utilization < 1 for r in rows)

    def test_cost(self):
        c = staffing_cost(3, 25, 20, 8, wait_cost=10)
        assert c["total_cost"] > 0
        assert c["server_cost"] == 75


class TestCalibration:
    def test_all_pass(self):
        from forgequeue.calibration import calibrate
        r = calibrate()
        for c in r["results"]:
            if not c["passed"]:
                print(f'FAIL: {c}')
        assert r["is_calibrated"]
