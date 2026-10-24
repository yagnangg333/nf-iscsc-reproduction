"""Tests for the beamforming problem (41) and Alg. 3 (Stage 5)."""
import numpy as np
import pytest

import allocation
import beamforming as bf
import channel
import config
import crb
import scenario


def _slot(k=5, seed=0, nr=config.NR):
    lay = scenario.place_vehicles(k, seed)
    q = [scenario.initial_states(lay, m) for m in range(config.M_RSU)]
    th = np.array([x[:, 0] for x in q])
    d = np.array([x[:, 1] for x in q])
    be = np.array([x[:, 3] for x in q])
    return bf.build_slot(th, d, be, nr=nr), d


def _random_beams(slot, xi, rng, power=0.5):
    W, R = {}, []
    for m in range(xi.shape[0]):
        n = slot.U[m].shape[1]
        for k in np.flatnonzero(xi[m]):
            v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            W[(m, k)] = np.outer(v, v.conj())
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        R.append(A @ A.conj().T / n)
    return bf._project(W, R, xi, np.full(xi.shape[0], power))


def test_lemma1_as_printed_has_the_wrong_sign():
    # Eq. (36) says ln|E^-1| = min_A Tr(AE) - ln|A| - N. The minimum (A = E^-1) is ln|E|.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    E = np.eye(3) + X @ X.conj().T
    A = np.linalg.inv(E)
    val = np.real(np.trace(A @ E)) - np.linalg.slogdet(A)[1] - 3
    assert val == pytest.approx(np.linalg.slogdet(E)[1])
    assert val != pytest.approx(-np.linalg.slogdet(E)[1])


def test_subspace_keeps_fim_exactly():
    # Any beam outside the subspace adds nothing to the FIM (16)-(18).
    slot, _ = _slot()
    rng = np.random.default_rng(1)
    m, k = 0, 2
    th, d, be = slot.theta[m, k], slot.d[m, k], slot.beta[m, k]
    X = rng.standard_normal((config.NT, 4)) + 1j * rng.standard_normal((config.NT, 4))
    U = slot.U[m]
    Xp = U @ (U.conj().T @ X)
    J_full = crb.fim_paper(th, d, be, X, config.SIGMA2_R_W / config.PT_W)      # Pt units, as in bf
    J_proj = crb.fim_paper(th, d, be, Xp, config.SIGMA2_R_W / config.PT_W)
    np.testing.assert_allclose(J_proj, J_full, rtol=1e-6)
    # and the reduced blocks give the same FIM
    J_red = bf.fim_from_x(slot.fim[m][k], (U.conj().T @ X) @ (U.conj().T @ X).conj().T)
    np.testing.assert_allclose(J_red, J_full, rtol=1e-6)


def test_subspace_keeps_channel_gain():
    slot, _ = _slot()
    m, k = 1, 3
    H = channel.channel_matrix(slot.theta[m, k], slot.d[m, k]).conj().T
    g = np.sqrt(config.PT_W / config.SIGMA2_C_W)
    np.testing.assert_allclose(slot.U[m] @ slot.Hs[m][k], g * H, atol=1e-6 * np.abs(g * H).max())


def test_rate_matches_eq13_in_full_space():
    slot, d = _slot(k=4, seed=2)
    xi = allocation.to_matrix(allocation.greedy(d))
    rng = np.random.default_rng(3)
    W, R = _random_beams(slot, xi, rng)
    rho = np.full(xi.shape, 0.9)
    s = bf.rates(slot, xi, W, R, rho, 1.1)
    # direct Eq. (13) with full Nt x Nt matrices, interference over each RSU's own channel
    full = lambda m, X: slot.U[m] @ X @ slot.U[m].conj().T * config.PT_W
    for k in range(xi.shape[1]):
        m = int(np.argmax(xi[:, k]))
        H = {mm: channel.channel_matrix(slot.theta[mm, k], slot.d[mm, k]).conj().T for mm in range(2)}
        sig = H[m].conj().T @ full(m, W[(m, k)]) @ H[m]
        intf = sum(H[mm].conj().T @ full(mm, R[mm]) @ H[mm] for mm in range(2))
        intf = intf + sum(H[mm].conj().T @ full(mm, W[(mm, kk)]) @ H[mm]
                          for mm in range(2) for kk in np.flatnonzero(xi[mm]) if kk != k)
        e = np.linalg.slogdet(np.eye(3) + (sig + intf) / config.SIGMA2_C_W)[1] \
            - np.linalg.slogdet(np.eye(3) + intf / config.SIGMA2_C_W)[1]
        assert s[k] == pytest.approx(1.1 / 0.9 * e / np.log(2), rel=1e-6)


def test_surrogate_gradient_finite_difference():
    slot, d = _slot()
    xi = allocation.to_matrix(allocation.greedy(d))
    rho = np.full(xi.shape, 0.9)
    W, R = bf.initial_beams(slot, xi, np.array([0.9, 0.9]))
    rng = np.random.default_rng(1)
    for eps in (0.5, 0.0):
        f, (gW, gR) = bf.surrogate(slot, xi, W, R, rho, 1.1, eps=eps, grad=True)
        for key in list(W)[:2] + ["R"]:
            X = R[1] if key == "R" else W[key]
            D = rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)
            D = (D + D.conj().T) / 2 * 1e-7
            Wp, Rp, Wm, Rm = dict(W), list(R), dict(W), list(R)
            if key == "R":
                Rp[1], Rm[1], g = R[1] + D, R[1] - D, gR[1]
            else:
                Wp[key], Wm[key], g = W[key] + D, W[key] - D, gW[key]
            fd = (bf.surrogate(slot, xi, Wp, Rp, rho, 1.1, eps=eps)
                  - bf.surrogate(slot, xi, Wm, Rm, rho, 1.1, eps=eps)) / 2
            assert fd == pytest.approx(np.real(np.trace(g @ D)), rel=1e-4, abs=1e-15)


def test_lemma2_bound_is_conservative():
    # eta_S from (39) never exceeds the true rate of Eq. (13) (log(1 + x) <= x).
    slot, d = _slot(k=6, seed=4)
    xi = allocation.to_matrix(allocation.greedy(d))
    rho = np.full(xi.shape, 0.81)
    res = bf.solve_beams(slot, xi, rho, np.full(2, 0.3), 1.1)
    exact = bf.rates(slot, xi, res["W"], res["R"], rho, 1.1) * np.log(2)
    assert np.all(res["eta_S"] <= exact + 1e-9)


def test_project_capped():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lam = rng.standard_normal(12)
        cap = rng.uniform(0.1, 3)
        mu = bf._project_capped(lam, cap)
        assert mu.min() >= 0 and mu.sum() <= cap + 1e-12
        # optimality: no other feasible point is closer (check against random feasible points)
        for _ in range(20):
            z = np.abs(rng.standard_normal(12))
            z *= rng.uniform(0, cap) / z.sum()
            assert np.linalg.norm(lam - mu) <= np.linalg.norm(lam - z) + 1e-12


def test_projection_is_feasible_and_idempotent():
    slot, d = _slot()
    xi = allocation.to_matrix(allocation.greedy(d))
    rng = np.random.default_rng(2)
    W, R = {}, []
    for m in range(2):
        n = slot.U[m].shape[1]
        for k in np.flatnonzero(xi[m]):
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            W[(m, k)] = A + A.conj().T
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        R.append(A + A.conj().T)
    budget = np.array([0.4, 0.7])
    Wp, Rp = bf._project(W, R, xi, budget)
    for m in range(2):
        tot = np.real(np.trace(Rp[m]) + sum(np.trace(Wp[(m, k)]) for k in np.flatnonzero(xi[m])))
        assert tot <= budget[m] + 1e-9
        assert np.linalg.eigvalsh(Rp[m]).min() > -1e-12
    W2, R2 = bf._project(Wp, Rp, xi, budget)
    for key in Wp:
        np.testing.assert_allclose(W2[key], Wp[key], atol=1e-10)


@pytest.mark.parametrize("solver", ["SPG", "LBFGS"])
def test_solver_matches_cvxpy_on_a_small_case(solver):
    # CVXPY (CLARABEL) only converges at low SNR here, so we scale the channel down.
    slot, d = _slot(k=2, seed=1)
    slot.Hs = [[h * 0.01 for h in hm] for hm in slot.Hs]
    xi = allocation.to_matrix([0, 1])
    rho = np.full(xi.shape, 0.9)
    pa = np.full(2, 0.3)
    a = bf.solve_beams(slot, xi, rho, pa, 1.1, solver=solver)
    b = bf.solve_beams(slot, xi, rho, pa, 1.1, solver="CLARABEL")
    if b is None:
        pytest.skip("CLARABEL did not converge on this machine")
    fa = bf.surrogate(slot, xi, a["W"], a["R"], rho, 1.1)
    fb = bf.surrogate(slot, xi, b["W"], b["R"], rho, 1.1)
    assert fa <= fb + 1e-4 * abs(fb)
    assert np.all(bf.beam_power(xi, a["W"], a["R"]) <= pa * (1 + 1e-6))      # (34e), in W


def test_inner_problem_respects_power_budget():
    slot, d = _slot(k=5, seed=0)
    xi = allocation.to_matrix(allocation.greedy(d))
    rng = np.random.default_rng(0)
    c, dbits = allocation.draw_tasks(5, rng)
    nu3 = allocation.draw_nu3(5, rng)
    for sem in (True, False):
        r = bf.inner_problem(slot, xi, c, dbits, nu3, semantic=sem, rng=np.random.default_rng(1),
                             latency="relaxed")
        tot = r["p_cs"] + r["p_comp"] + r["p_dt"]
        assert np.all(tot <= config.PT_W * (1 + 1e-6))            # (34e)
        assert np.all(r["rho"][xi > 0] >= config.RHO_LB - 1e-12)  # (34i)
        if not sem:
            assert np.all(r["rho"] == 1.0) and np.all(r["p_comp"] == 0)
        # rank one after randomisation
        for X in r["W"].values():
            w = np.linalg.eigvalsh(X)
            assert w[-2] <= 1e-9 * w[-1]


def test_infeasible_when_cpu_too_slow():
    slot, d = _slot(k=5, seed=0)
    xi = allocation.to_matrix(allocation.greedy(d))
    c, dbits = np.full(5, 2000.0), np.full(5, 3000.0)
    info = {}
    assert bf.inner_problem(slot, xi, c, dbits, np.full(5, 1e9), latency="relaxed", info=info) is None
    assert info["reason"] == "cpu"


def test_written_latency_fails_for_greedy_k5():
    # Finding of Stage 5: with Table III's nu1, nu2, L of Eq. (24) breaks (34h) at K = 5.
    slot, d = _slot(k=5, seed=0)
    xi = allocation.to_matrix(allocation.greedy(d))
    rng = np.random.default_rng([0, 5, 5])
    c, dbits = allocation.draw_tasks(5, rng)
    nu3 = allocation.draw_nu3(5, rng)
    info = {}
    assert bf.inner_problem(slot, xi, c, dbits, nu3, info=info) is None
    assert info["reason"] == "cpu"
