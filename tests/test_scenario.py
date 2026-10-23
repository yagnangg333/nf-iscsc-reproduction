"""Checks for scenario.py and the motion model (tracking/models.py)."""
import numpy as np
import pytest

import config
import scenario
from tracking import models

KS = (1, 5, 10, 15, 20)


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("seed", range(20))
def test_spacing_and_bounds(k, seed):
    lay = scenario.place_vehicles(k, seed)
    assert lay.x.size == k
    # every car fully inside the 100 m x 10 m segment
    assert np.all(lay.x - config.VEH_LENGTH / 2 >= -1e-9)
    assert np.all(lay.x + config.VEH_LENGTH / 2 <= config.ROAD_LENGTH + 1e-9)
    assert np.all(lay.y - config.VEH_WIDTH / 2 >= 0)
    assert np.all(lay.y + config.VEH_WIDTH / 2 <= config.ROAD_WIDTH)
    # no two cars in the same lane closer than 3.5 m bumper to bumper
    assert scenario.min_bumper_gap(lay.x, lay.lane) >= config.MIN_GAP - 1e-9


def test_same_seed_same_layout():
    a, b = scenario.place_vehicles(10, 7), scenario.place_vehicles(10, 7)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.lane, b.lane)
    c = scenario.place_vehicles(10, 8)
    assert not np.array_equal(a.x, c.x)


def test_capacity():
    # 95.5 m of centre positions, 8 m spacing -> 12 cars per lane, 24 in total
    assert scenario.max_per_lane() == 12
    scenario.place_vehicles(24, 0)
    with pytest.raises(ValueError):
        scenario.place_vehicles(25, 0)


def test_acceptance_formula_matches_monte_carlo():
    # Analytic chance that a pure uniform layout meets the rule vs brute force.
    rng = np.random.default_rng(1)
    k, n = 6, 20000
    ok = sum(scenario.min_bumper_gap(*scenario.naive_uniform(k, rng)) >= config.MIN_GAP
             for _ in range(n))
    p = scenario.acceptance_probability(k)
    assert ok / n == pytest.approx(p, abs=4 * np.sqrt(p * (1 - p) / n))


def test_layout_law_matches_rejection_sampling():
    # Our sampler vs plain rejection sampling: compare mean lane-0 count and
    # mean smallest gap for K = 6 (cheap enough to reject).
    rng = np.random.default_rng(2)
    rej_n0, rej_gap = [], []
    while len(rej_n0) < 3000:
        x, lane = scenario.naive_uniform(6, rng)
        g = scenario.min_bumper_gap(x, lane)
        if g >= config.MIN_GAP:
            rej_n0.append(np.sum(lane == 0))
            rej_gap.append(g)
    ours = [scenario.place_vehicles(6, s) for s in range(3000)]
    our_n0 = [np.sum(l.lane == 0) for l in ours]
    our_gap = [scenario.min_bumper_gap(l.x, l.lane) for l in ours]
    assert np.mean(our_n0) == pytest.approx(np.mean(rej_n0), abs=0.1)
    assert np.mean(our_gap) == pytest.approx(np.mean(rej_gap), rel=0.05)


def test_polar_round_trip():
    lay = scenario.place_vehicles(10, 3)
    for m in range(config.M_RSU):
        th, d = scenario.to_polar(lay.x, lay.y, m)
        rx, ry = config.RSU_POSITIONS[m]
        np.testing.assert_allclose(rx + d * np.cos(th), lay.x, atol=1e-9)
        np.testing.assert_allclose(ry + d * np.sin(th), lay.y, atol=1e-9)


@pytest.mark.parametrize("m", range(config.M_RSU))
def test_eq10_one_step_matches_straight_line_motion(m):
    # Eq. (10) is a first-order step of constant-velocity motion towards -x.
    # Over one 20 ms slot it should match the exact geometry to second order.
    lay = scenario.place_vehicles(10, 4)
    q0 = scenario.initial_states(lay, m)
    q1 = models.g1(q0)
    x1, y1 = scenario.move(lay, config.DELTA_T)
    th1, d1 = scenario.to_polar(x1, y1, m)
    step = config.VEH_SPEED * config.DELTA_T                # 0.4 m
    np.testing.assert_allclose(q1[:, 1], d1, atol=step**2 / q0[:, 1].min())
    np.testing.assert_allclose(q1[:, 0], th1, atol=step**2 / q0[:, 1].min() ** 2)


def test_eq10_zero_speed_is_static():
    q0 = np.array([0.7, 30.0, 0.0, 1e-4])
    np.testing.assert_allclose(models.propagate(q0, 50)[-1], q0)


def test_eq10_beta_follows_one_over_d():
    # beta update (1 + v dT cos(theta)/d) is the first-order change of 1/d,
    # so beta * d stays almost constant (Euler drift ~0.2% per second; a 1/d^2
    # law would change it by ~30% here).
    q = models.propagate(np.array([0.5, 60.0, 20.0, 1.0 / 60.0]), 50)
    np.testing.assert_allclose(q[:, 3] * q[:, 1], 1.0, rtol=5e-3)
