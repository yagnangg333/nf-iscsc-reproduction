"""Baseline from [18]: Z. Wang, X. Mu, Y. Liu, "Near-field integrated sensing and communications",
IEEE Commun. Lett., vol. 27, no. 8, 2023 ("Benchmark with Nr = 1" in Fig. 5 of the main paper).

[18] problem (16), fully digital antennas:
    min  tr(CRB(R_x))                         (16a)
    s.t. R(f_k, R_x) >= R_min,k  for all k    (16b), rate of Eq. (11), single-antenna users
         tr(R_x) <= P_max                     (16c)
         R_x >= sum_k f_k f_k^H               (16d)
solved by SDR, problem (20): F_k = f_k f_k^H >= 0, the rate constraint as the linear form (18)
    gamma_k h_k^T F_k h_k^* >= h_k^T R_x h_k^* + sigma_k^2,   gamma_k = 1 + 1 / (2^R_min,k - 1),
and the CRB through the LMI (17b) with an auxiliary 2 x 2 matrix U. Rank one is rebuilt by Eq. (21):
    f_k = (h_k^T F_k h_k^*)^(-1/2) F_k h_k^*,   R_x unchanged.

What we had to add to put [18] (one BS, one target) into the main paper's setting (docs/stage6-notes.md):
  - Two RSUs solved jointly. A vehicle's interference term in (18) sums h^H R_x h over both RSUs,
    each over its own channel (same as our deviation 1 in Stage 5). Still linear, so still convex.
  - Every served vehicle is both a user and a target. tr(CRB) is summed over targets, each with its own
    FIM (the main paper's eta). [18] footnote 1 says the multi-target CRB has the same form.
  - CRB units: CRB(d) in m^2 plus CRB(theta) in rad^2, as in the main paper's objective.
  - P_max per RSU = Pt - P_DT (no semantic extraction, so no P_Comp).
  - R_min,k is not given for the benchmark in the main paper. It is a free input here (swept).
Channels and FIM blocks come from beamforming.build_slot with nr = 1, in the same reduced subspace.

Solvers. solve_benchmark18 (CVXPY, CLARABEL) is the SDP exactly as written, but it is only accurate on
small cases: tr(CRB) spans about 1e-10 to 1e2 here, and at K = 20 CVXPY runs out of memory. So the runs
use solve_benchmark18_lbfgs: the same problem, with the CRB part of beamforming.surrogate (eps = 0, which
is exactly tr(CRB)), the factorised L-BFGS solver of Stage 5, and (18) handled by an augmented Lagrangian.
The two agree on small cases (tests/test_baselines.py).
"""
import cvxpy as cp
import numpy as np

import beamforming as bf
import config


def _tr(B, X):
    """Re tr(B X) for a numeric B and a CVXPY Hermitian X, without forming the n x n product."""
    return cp.real(cp.sum(cp.multiply(B.T, X)))


def solve_benchmark18(slot, xi, p_avail, r_min, solver="CLARABEL"):
    """[18] problem (20) for all served pairs of both RSUs, then Eq. (21).

    slot: beamforming.SlotData with nr = 1. xi: (M, K) assignment. p_avail: (M,) power for beams, W.
    r_min: minimum rate per vehicle, bit/s/Hz (scalar or (K,)).
    Returns dict(W, R, value, status) in the reduced, Pt-normalised units of beamforming.py, or None.
    """
    assert slot.nr == 1, "[18] has single-antenna users"
    M, K = xi.shape
    r_min = np.broadcast_to(np.asarray(r_min, float), (K,))
    F = {(m, k): cp.Variable((slot.U[m].shape[1],) * 2, hermitian=True)
         for m in range(M) for k in np.flatnonzero(xi[m])}
    # R_x is the variable and (19) is R_x - sum_k F_k >= 0, as in [18]. (Writing R_x = R_s + sum F_k
    # instead makes every FIM term touch K + 1 variables, which runs out of memory at K = 20.)
    Rx = [cp.Variable((slot.U[m].shape[1],) * 2, hermitian=True) for m in range(M)]
    cons = [X >> 0 for X in F.values()]
    cons += [Rx[m] - sum(F[(m, k)] for k in np.flatnonzero(xi[m])) >> 0 for m in range(M)]   # (19)
    obj = 0
    for m in range(M):
        cons.append(cp.real(cp.trace(Rx[m])) <= p_avail[m] / config.PT_W)  # (16c)
        for k in np.flatnonzero(xi[m]):
            # (18). Hs is already scaled by sqrt(Pt / sigma_c^2), so the noise term is 1.
            gamma = 1 + 1 / (2 ** r_min[k] - 1)
            h = slot.Hs[m][k][:, 0]
            sig = _tr(np.outer(h, h.conj()), F[(m, k)])                       # h^H F h
            tot = sum(_tr(np.outer(slot.Hs[mm][k][:, 0], slot.Hs[mm][k][:, 0].conj()), Rx[mm]) for mm in range(M))
            cons.append((gamma * sig - tot - 1) / np.real(h.conj() @ h) >= 0)   # divided by |h|^2 for scaling
            # (17b): [J11 - U, J12; J12^T, J22] >= 0, scaled as in beamforming.solve_beams_cvx.
            b, D = slot.fim[m][k], slot.scale[m][k]
            J = [[None] * 4 for _ in range(4)]
            for a in range(2):
                for c in range(a, 2):
                    J[a][c] = J[c][a] = D[a] * D[c] * _tr(b["M"][a][c], Rx[m])
                J[a][2] = J[2][a] = D[a] * D[2] * _tr(b["C"][a], Rx[m])
                J[a][3] = J[3][a] = D[a] * D[3] * _tr(1j * b["C"][a], Rx[m])
            J[2][2] = J[3][3] = D[2] ** 2 * _tr(b["G"], Rx[m])
            J[2][3] = J[3][2] = cp.Constant(0.0)
            u = cp.Variable((2, 2), symmetric=True)
            lmi = cp.bmat([[J[0][0] - u[0, 0], J[0][1] - u[0, 1], J[0][2], J[0][3]],
                           [J[1][0] - u[1, 0], J[1][1] - u[1, 1], J[1][2], J[1][3]],
                           [J[2][0], J[2][1], J[2][2], J[2][3]],
                           [J[3][0], J[3][1], J[3][2], J[3][3]]])
            cons += [(lmi + lmi.T) / 2 >> 0, u >> 0]
            obj += cp.matrix_frac(np.diag(D[:2]), u)                        # (17a): tr(U^-1) in SI units
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=solver)
    except cp.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    Fv = {key: bf._psd(v.value) for key, v in F.items()}
    Rxv = [bf._psd(Rx[m].value) for m in range(M)]
    W, R = rank_one(slot, xi, Fv, Rxv)
    return dict(W=W, R=R, F=Fv, Rx=Rxv, value=prob.value, status=prob.status)


def rank_one(slot, xi, F, Rx):
    """Eq. (21): f_k = (h^H F_k h)^(-1/2) F_k h; R_x is kept, so R_s = R_x - sum f_k f_k^H."""
    W, R = {}, []
    for m in range(xi.shape[0]):
        rest = Rx[m].copy()
        for k in np.flatnonzero(xi[m]):
            h = slot.Hs[m][k][:, 0]
            g = np.real(h.conj() @ F[(m, k)] @ h)
            f = F[(m, k)] @ h / np.sqrt(max(g, 1e-300))
            W[(m, k)] = np.outer(f, f.conj())
            rest = rest - W[(m, k)]
        R.append(bf._psd(rest))
    return W, R


def _rate_terms(slot, xi, W, R):
    """s_k = h^H F_k h and t_k = sum over RSUs of h^H R_x h, for each vehicle (noise = 1)."""
    M, K = xi.shape
    serve = np.argmax(xi, axis=0)
    Rx = [R[m] + sum((W[(m, k)] for k in np.flatnonzero(xi[m])), np.zeros_like(R[m])) for m in range(M)]
    s = np.array([np.real(slot.Hs[serve[k]][k][:, 0].conj() @ W[(serve[k], k)] @ slot.Hs[serve[k]][k][:, 0])
                  for k in range(K)])
    t = np.array([sum(np.real(slot.Hs[m][k][:, 0].conj() @ Rx[m] @ slot.Hs[m][k][:, 0]) for m in range(M))
                  for k in range(K)])
    return s, t


def solve_benchmark18_lbfgs(slot, xi, p_avail, r_min, outer=40, tol=1e-4, max_iter=1000):
    """[18] problem (16)/(20) by augmented Lagrangian + factorised L-BFGS, then Eq. (21).

    The rate constraint (18), gamma s >= t + 1, is the same set as g_k = SINR_k / (2^R_min - 1) - 1 >= 0
    with SINR = s / (t - s + 1). We use g: a small error in g is a small relative error in SINR, while
    the same error in gamma s / (t + 1) - 1 costs a lot of rate when R_min is high (gamma close to 1).
    Returns dict(W, R, F, Rx, max_violation, status) or None if (18) cannot be met.
    """
    assert slot.nr == 1, "[18] has single-antenna users"
    M, K = xi.shape
    q = 2 ** np.broadcast_to(np.asarray(r_min, float), (K,)) - 1          # SINR target, 2^R_min - 1
    serve = np.argmax(xi, axis=0)
    hh = [[np.outer(slot.Hs[m][k][:, 0], slot.Hs[m][k][:, 0].conj()) for k in range(K)] for m in range(M)]
    ones = np.ones(xi.shape)
    lam, mu = np.zeros(K), 10.0

    def g_of(W, R):
        s, t = _rate_terms(slot, xi, W, R)
        return s / (t - s + 1) / q - 1, s, t

    def extra(W, R):
        g, s, t = g_of(W, R)
        # AL term for g >= 0: (mu/2) max(0, lam/mu - g)^2 - lam^2 / (2 mu); its slope in g is -max(0, lam - mu g).
        a = np.maximum(0.0, lam / mu - g)
        f = float(np.sum(mu / 2 * a ** 2 - lam ** 2 / (2 * mu)))
        dg = -mu * a                                              # d f / d g_k
        gW = {key: np.zeros_like(v) for key, v in W.items()}
        gR = [np.zeros_like(r) for r in R]
        for k in range(K):
            if dg[k] == 0:
                continue
            u = t[k] - s[k] + 1                                    # interference + noise
            ds = dg[k] * (1 / u + s[k] / u ** 2) / q[k]            # d f / d s_k
            dt = -dg[k] * s[k] / u ** 2 / q[k]                     # d f / d t_k
            gW[(serve[k], k)] += ds * hh[serve[k]][k]
            for m in range(M):                                     # t_k depends on every matrix of every RSU
                gR[m] += dt * hh[m][k]
                for kk in np.flatnonzero(xi[m]):
                    gW[(m, kk)] += dt * hh[m][k]
        return f, gW, gR

    start, viol = None, np.inf
    for it in range(outer):
        res = bf.solve_beams_lbfgs(slot, xi, ones, p_avail, 1.0, eps=0.0, start=start, max_iter=max_iter,
                                   extra=extra)
        if res is None:
            return None
        start = (res["W"], res["R"])
        g = g_of(res["W"], res["R"])[0]
        viol = float(np.max(np.maximum(-g, 0)))
        lam = np.maximum(0.0, lam - mu * g)
        if viol < tol and np.all(lam * np.abs(g) < 1e-3 * (1 + np.abs(lam))):
            break
        if viol > tol:
            mu *= 4
    Rx = [res["R"][m] + sum((res["W"][(m, k)] for k in np.flatnonzero(xi[m])), np.zeros_like(res["R"][m]))
          for m in range(M)]
    W, R = rank_one(slot, xi, res["W"], Rx)
    return dict(W=W, R=R, F=res["W"], Rx=Rx, max_violation=viol, status=f"al_{it + 1}_rounds")
