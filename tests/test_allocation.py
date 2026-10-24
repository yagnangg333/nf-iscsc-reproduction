"""Tests for the computing, power and assignment models (Stage 5)."""
import numpy as np
import pytest

import allocation
import config


def test_comp_power_eq20_units():
    # Eq. (20): -F sum ln(rho). F = 10 mW, rho = 0.81 -> 2.107 mW per vehicle.
    assert allocation.comp_power([0.81], "mW") == pytest.approx(-10e-3 * np.log(0.81))
    assert allocation.comp_power([0.81, 0.81], "W") == pytest.approx(-20 * np.log(0.81))
    assert allocation.comp_power([1.0, 1.0]) == 0.0


def test_dt_power_readings():
    f, c, d = 0.4e9, 2000.0, 3000.0
    assert allocation.dt_power(f, c, d, "as_written") == pytest.approx(1e-28 * f**3 * c)          # 12.8 W
    assert allocation.dt_power(f, c, d, "energy_per_slot") == pytest.approx(1e-28 * f**2 * c * d / 0.02)
    assert allocation.dt_power(f, c, d, "kappa_f3") == pytest.approx(1e-28 * f**3)
    with pytest.raises(ValueError):
        allocation.dt_power(f, c, d, "other")


def test_energy_reading_matches_fig9_shape():
    # Fig. 9: power grows as f^2 and in proportion to the data size.
    f = np.array([1, 2, 3, 4, 5, 6]) * 1e9
    p = allocation.dt_power(f, 10.0, 1000.0, "energy_per_slot")
    assert np.polyfit(np.log(f), np.log(p), 1)[0] == pytest.approx(2.0)
    assert allocation.dt_power(1e9, 10.0, 1e4, "energy_per_slot") == pytest.approx(
        10 * allocation.dt_power(1e9, 10.0, 1e3, "energy_per_slot"))


def test_min_frequency_meets_latency_exactly():
    c, d, l_extra = 1500.0, 2000.0, 1e6
    f = allocation.min_frequency(c, d, l_extra)
    assert allocation.latency(c, d, l_extra, f) == pytest.approx(config.T_MAX)
    assert allocation.latency(c, d, l_extra, 1.01 * f) < config.T_MAX


def test_workload_linear_eq24():
    assert allocation.workload(0.0, 0.0, 30.0) == pytest.approx(30.0)
    assert allocation.workload(1.0, 0.0, 0.0) == pytest.approx(243.2e6)     # per degree
    assert allocation.workload(0.0, 1.0, 0.0) == pytest.approx(121.6e6)     # per metre
    assert allocation.workload(0.0, 0.0, -30.0) == pytest.approx(30.0)      # |nu3|


def test_computing_plan_only_served_and_fmax():
    xi = allocation.to_matrix([0, 0, 1])
    c = np.full(3, 1000.0)
    d = np.full(3, 1000.0)
    f, p_dt, ok = allocation.computing_plan(xi, c, d, np.zeros(3))
    assert f[0, 2] == 0 and f[1, 0] == 0 and f[1, 1] == 0
    assert f[0, 0] == pytest.approx(1e6 / config.T_MAX)
    assert np.all(ok)
    # a huge workload breaks (34h)
    f, p_dt, ok = allocation.computing_plan(xi, c, d, np.array([1e9, 0, 0]))
    assert not ok[0] and ok[1]


def test_draws_seeded_and_in_range():
    a = allocation.draw_tasks(50, np.random.default_rng(3))
    b = allocation.draw_tasks(50, np.random.default_rng(3))
    np.testing.assert_array_equal(a[0], b[0])
    assert a[0].min() >= 1e3 and a[0].max() <= 2e3
    assert a[1].min() >= 1e3 and a[1].max() <= 3e3
    nu = allocation.draw_nu3(20000, np.random.default_rng(0))
    assert np.mean(nu**2) == pytest.approx(1000, rel=0.05)      # E|nu3|^2 = 1000
    assert nu.min() >= 0


def test_greedy_nearest_and_matrix():
    dist = np.array([[10.0, 50.0, 30.0], [40.0, 20.0, 31.0]])
    a = allocation.greedy(dist)
    np.testing.assert_array_equal(a, [0, 1, 0])
    xi = allocation.to_matrix(a)
    np.testing.assert_array_equal(xi.sum(axis=0), 1)            # (34f)


def test_random_flip_extremes():
    a = np.array([0, 1, 0, 1])
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(allocation.random_flip(a, rng, p=0.0), a)
    np.testing.assert_array_equal(allocation.random_flip(a, rng, p=1.0), 1 - a)
    n = allocation.neighbour(a, rng)
    assert np.sum(n != a) == 1


def test_sa_finds_best_of_toy_problem():
    # Score = number of vehicles on RSU 1, minus a penalty if all are on RSU 1; best has K-1 on RSU 1.
    k = 6
    dist = np.vstack([np.ones(k), 2 * np.ones(k)])                # greedy puts all on RSU 0

    def solve(a):
        s = float(np.sum(a == 1))
        return s - 10.0 if s == k else s

    best, b_best, hist = allocation.hybrid_heuristic(dist, solve, np.random.default_rng(1))
    assert b_best == k - 1
    assert len(hist) <= config.SA_NMAX
    assert all(h[2] >= h[1] - 1e-12 for h in hist)                # best never below current


def test_sa_skips_infeasible():
    k = 4
    dist = np.vstack([np.ones(k), 2 * np.ones(k)])

    def solve(a):
        return None if a.sum() >= 3 else float(a.sum())          # 3 or 4 on RSU 1 is infeasible

    best, b_best, _ = allocation.hybrid_heuristic(dist, solve, np.random.default_rng(0))
    assert b_best == 2 and best.sum() == 2


def test_fig8_operating_point():
    # Fig. 8 (linear model): about 10 GHz for 100 Kb and 1 GHz for 10 Kb at T = 0.01 s, with C = 1000.
    assert allocation.min_frequency(1000.0, 1e5, 0.0, 0.01) == pytest.approx(10e9)
    assert allocation.min_frequency(1000.0, 1e4, 0.0, 0.01) == pytest.approx(1e9)


def test_100kb_task_exceeds_fmax_alone():
    # Stage 6: with Table III's lowest C, one 100 Kb task needs 6.7 GHz in 15 ms, above F^max = 5.8 GHz.
    f = allocation.min_frequency(config.C_CYCLES_RANGE[0], 1e5, 0.0)
    assert f == pytest.approx(1e3 * 1e5 / config.T_MAX)
    assert f > config.F_MAX
    # 10 Kb fits for one vehicle, but not for 10 vehicles on one RSU at the highest C.
    assert allocation.min_frequency(config.C_CYCLES_RANGE[1], 1e4, 0.0) < config.F_MAX
    assert 10 * allocation.min_frequency(config.C_CYCLES_RANGE[1], 1e4, 0.0) > config.F_MAX
