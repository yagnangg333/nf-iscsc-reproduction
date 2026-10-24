"""Tests for the [18] benchmark (baselines.py, Stage 6)."""
import numpy as np
import pytest

import allocation
import baselines
import beamforming as bf
import config
import scenario


def _slot(k, seed, scale=1.0):
    lay = scenario.place_vehicles(k, seed)
    q = [scenario.initial_states(lay, m) for m in range(config.M_RSU)]
    th, d, be = (np.array([x[:, j] for x in q]) for j in (0, 1, 3))
    slot = bf.build_slot(th, d, be, nr=1)
    slot.Hs = [[h * scale for h in hm] for hm in slot.Hs]
    return slot, d


def _sum_crb(slot, xi, W, R):
    cd, ct = bf.crbs(slot, xi, W, R)
    return float(np.sum(cd + ct))


def test_gamma_form_of_rate_constraint():
    # [18] Eq. (18): gamma s >= t + 1 with gamma = 1 + 1 / (2^R - 1) is the same as log2(1 + SINR) >= R.
    r_min = 3.0
    gamma = 1 + 1 / (2 ** r_min - 1)
    s = 5.0
    for interf in (0.0, 0.3, 2.0):
        t = s + interf                       # t includes the user's own signal
        lhs = gamma * s >= t + 1
        assert lhs == (np.log2(1 + s / (interf + 1)) >= r_min - 1e-12)


def test_rank_one_rebuild_keeps_signal_and_rx():
    # Eq. (21): h^H f f^H h = h^H F h, and R_x is unchanged.
    slot, d = _slot(3, 0)
    xi = allocation.to_matrix(allocation.greedy(d))
    rng = np.random.default_rng(0)
    F, Rx = {}, []
    for m in range(2):
        n = slot.U[m].shape[1]
        tot = np.zeros((n, n), complex)
        for k in np.flatnonzero(xi[m]):
            A = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
            F[(m, k)] = A @ A.conj().T
            tot += F[(m, k)]
        Rx.append(tot + np.eye(n))
    W, R = baselines.rank_one(slot, xi, F, Rx)
    for (m, k), X in W.items():
        h = slot.Hs[m][k][:, 0]
        assert np.real(h.conj() @ X @ h) == pytest.approx(np.real(h.conj() @ F[(m, k)] @ h))
        assert np.linalg.matrix_rank(X, tol=1e-9 * np.abs(X).max()) == 1
    for m in range(2):
        tot = R[m] + sum(W[(m, k)] for k in np.flatnonzero(xi[m]))
        np.testing.assert_allclose(tot, Rx[m], atol=1e-9)
        assert np.linalg.eigvalsh(R[m]).min() > -1e-9


@pytest.mark.parametrize("k,seed,r_min", [(2, 1, 1.0), (3, 0, 1.0), (3, 0, 3.0)])
def test_lbfgs_meets_rates_and_matches_cvxpy(k, seed, r_min):
    # Cases without a far end-fire car, where CLARABEL reports "optimal". Problem (20) is convex,
    # so both should reach the same tr(CRB).
    slot, d = _slot(k, seed)
    xi = allocation.to_matrix(allocation.greedy(d))
    pa = np.full(2, 0.3)
    a = baselines.solve_benchmark18_lbfgs(slot, xi, pa, r_min)
    b = baselines.solve_benchmark18(slot, xi, pa, r_min)
    assert a is not None
    rate = bf.rates(slot, xi, a["W"], a["R"], np.ones(xi.shape), 1.0)
    assert rate.min() >= r_min - 0.01                                     # (16b) after Eq. (21)
    assert np.all(bf.beam_power(xi, a["W"], a["R"]) <= pa * (1 + 1e-6))  # (16c)
    if b is None or b["status"] != "optimal":
        pytest.skip("CLARABEL not accurate on this machine")
    fa, fb = _sum_crb(slot, xi, a["W"], a["R"]), _sum_crb(slot, xi, b["W"], b["R"])
    assert fa == pytest.approx(fb, rel=5e-3)


def test_higher_rate_never_lowers_crb():
    # The sensing-communication trade-off of [18] Fig. 2: tr(CRB) is non-decreasing in R_min.
    slot, d = _slot(4, 2)
    xi = allocation.to_matrix(allocation.greedy(d))
    pa = np.full(2, 0.3)
    vals = [_sum_crb(slot, xi, *(lambda r: (r["W"], r["R"]))(baselines.solve_benchmark18_lbfgs(slot, xi, pa, q)))
            for q in (0.5, 4.0, 8.0)]
    assert vals[0] <= vals[1] * (1 + 1e-3) and vals[1] <= vals[2] * (1 + 1e-3)


def test_al_gradient_matches_finite_difference():
    # Check the hand-written gradient of the rate-constraint term through a full solve's extra() hook.
    slot, d = _slot(3, 0)
    xi = allocation.to_matrix(allocation.greedy(d))
    pa = np.full(2, 0.3)
    captured = {}
    orig = bf.solve_beams_lbfgs

    def spy(*a, **kw):
        captured["extra"] = kw["extra"]
        return orig(*a, **kw)

    bf.solve_beams_lbfgs = spy
    try:
        r = baselines.solve_benchmark18_lbfgs(slot, xi, pa, 16.0, outer=2, max_iter=50)
    finally:
        bf.solve_beams_lbfgs = orig
    extra = captured["extra"]
    W, R = r["F"], [r["Rx"][m] - sum(r["F"][(m, k)] for k in np.flatnonzero(xi[m])) for m in range(2)]
    f, gW, gR = extra(W, R)
    rng = np.random.default_rng(0)
    key = next(iter(W))
    D = rng.standard_normal(W[key].shape) + 1j * rng.standard_normal(W[key].shape)
    D = (D + D.conj().T) / 2 * 1e-6 * np.abs(W[key]).max()
    Wp, Wm = dict(W), dict(W)
    Wp[key], Wm[key] = W[key] + D, W[key] - D
    fd = (extra(Wp, R)[0] - extra(Wm, R)[0]) / 2
    assert f > 0                                    # R_min = 16 is not met after 2 rounds, so the term is active
    assert fd == pytest.approx(np.real(np.trace(gW[key] @ D)), rel=1e-4)
