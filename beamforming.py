"""Beamforming, semantic ratio and the inner problem (41) (Stage 5, Alg. 3).

For a fixed vehicle assignment xi this solves
  min  sum_{m,k served} eps (-eta_S) + (1 - eps) eta                              (41a)
  s.t. power (34e), CRB LMIs (40), rate bound (39), rho_LB <= rho <= 1 (34i)
with W_{m,k} (one per served vehicle) and R_m >= 0 (Hermitian), then Gaussian
randomisation for rank-one beams (Alg. 3, line 8).

Deviations from the paper (all in docs/stage5-notes.md):
  1. Lemma 1 (Eq. 36) has the wrong sign: min_A Tr(AE) - ln|A| - N equals ln|E|, not
     ln|E^-1| (put A = E^-1). So (37)/(39) do not bound -S. The term -ln|E| is already
     convex, so we keep it exactly (cvxpy log_det). The auxiliary A (Alg. 3 line 4) is then
     not needed. Lemma 2 (Eq. 38) for the interference term is kept as written.
  2. Interference from RSU m' reaches vehicle k over RSU m''s own channel H_{m',k}. Eq. (13)
     writes H_{m,k} for every RSU, which mixes the antenna spaces of two different arrays.
  3. CRB terms and LMIs only for served pairs (xi = 1); Rx_m = sum_k W_{m,k} + R_m (Eq. 3).
     R_m is one PSD matrix (the paper's sum_k r r^H); rank-one only for W.
  4. W and R are solved in the subspace spanned by the channels, steering matrices and
     their derivatives of all vehicles at that RSU. Every term of (41) depends on W, R
     only through that subspace, so this is exact (up to a 1e-9 singular-value cut).
  5. rho step (Alg. 3 line 5): one rho per RSU, found by bisection on the slope of the
     objective with the beams rescaled to the power left after P_Comp(rho).

Units: X (reduced W or R) is in units of Pt, so Tr(X) = 1 is the full budget.
Rates in nats inside, reported in bit/s/Hz (log2).
"""
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

import allocation
import channel
import config
import crb

LN2 = np.log(2.0)


# --------------------------------------------------------------- slot geometry
@dataclass
class SlotData:
    """Everything (41) needs about one slot, for M RSUs and K vehicles."""
    theta: np.ndarray            # (M, K) rad
    d: np.ndarray                # (M, K) m
    beta: np.ndarray             # (M, K)
    nr: int
    U: list                      # per RSU: (Nt, n_m) orthonormal basis
    Hs: list                     # Hs[m][k]: (n_m, Nr) reduced channel, scaled by sqrt(Pt / sigma_c^2)
    fim: list                    # fim[m][k]: dict of reduced FIM matrices (Pt units)
    scale: list = field(default_factory=list)   # scale[m][k]: (4,) diag scaling for the LMI
    _cache: dict = field(default_factory=dict, repr=False)


def _stacks(slot):
    """Per RSU linear maps from vec(T) to every quantity the surrogate needs (cached).

    Lr: rows (k, r, s) give (H_k^H T H_k)[r, s]; Lc: rows (k, j) give Tr(N_kj T) for
    N = M00, M01, M10, M11, C0, C1, G. So each evaluation is two matrix products.
    """
    key = tuple(id(h) for hm in slot.Hs for h in hm)
    if slot._cache.get("key") != key:
        st = []
        for m in range(len(slot.U)):
            H = np.stack(slot.Hs[m])                                        # (K, n, Nr)
            K, n, nr = H.shape
            Lr = np.einsum("kir,kjs->krsij", H.conj(), H).reshape(K * nr * nr, n * n)
            N = []
            for b in slot.fim[m]:
                N += [b["M"][0][0], b["M"][0][1], b["M"][1][0], b["M"][1][1], b["C"][0], b["C"][1], b["G"]]
            Lc = np.stack([x.T.ravel() for x in N])                         # (7K, n^2)
            Q = np.einsum("kir,kjr->kij", H, H.conj())
            st.append(dict(H=H, Lr=Lr, Lc=Lc, Q=Q, n=n, K=K))
        slot._cache.clear()
        slot._cache.update(key=key, st=st)
    return slot._cache["st"]


def surrogate(slot, xi, W, R, rho, iota, eps=config.EPS_WEIGHT, grad=False):
    """Fast version of surrogate_reference (same value and gradient), using _stacks."""
    M, K = xi.shape
    st = _stacks(slot)
    nr = slot.nr
    serve = np.argmax(xi, axis=0)
    T = [R[m] + sum((W[(m, k)] for k in np.flatnonzero(xi[m])), np.zeros_like(R[m])) for m in range(M)]
    HTH = [(st[m]["Lr"] @ T[m].ravel()).reshape(K, nr, nr) for m in range(M)]
    E = np.eye(nr)[None] + sum(HTH)
    tr_int = sum(np.real(np.trace(x, axis1=1, axis2=2)) for x in HTH)
    for k in range(K):
        H = slot.Hs[serve[k]][k]
        tr_int[k] -= np.real(np.trace(H.conj().T @ W[(serve[k], k)] @ H))
    ck = eps * iota / rho[serve, np.arange(K)]
    ld = np.linalg.slogdet(E)[1]
    f = float(np.sum(-ck * (ld - tr_int)))
    crb_w = [None] * M
    for m in range(M):
        ks = np.flatnonzero(xi[m])
        if ks.size == 0:
            continue
        rows = (7 * ks[:, None] + np.arange(7)[None]).ravel()
        y = (st[m]["Lc"][rows] @ T[m].ravel()).reshape(ks.size, 7)
        J = np.zeros((ks.size, 4, 4))
        J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1] = (np.real(y[:, i]) for i in range(4))
        for a in range(2):
            J[:, a, 2] = J[:, 2, a] = np.real(y[:, 4 + a])
            J[:, a, 3] = J[:, 3, a] = np.real(1j * y[:, 4 + a])
        J[:, 2, 2] = J[:, 3, 3] = np.real(y[:, 6])
        try:
            Jinv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            return (np.inf, None) if grad else np.inf
        dg = Jinv[:, [0, 1], [0, 1]]
        if not np.all(np.isfinite(Jinv)) or np.any(dg <= 0):
            return (np.inf, None) if grad else np.inf
        f += (1 - eps) * float(np.sum(dg))
        crb_w[m] = (rows, Jinv)
    if not grad:
        return f
    Einv = np.linalg.inv(E)
    g_common = []
    for m in range(M):
        n = st[m]["n"]
        coef = (-ck[:, None, None] * np.conj(Einv)).ravel()
        g = np.conj((coef @ st[m]["Lr"]).reshape(n, n)) + np.einsum("k,kij->ij", ck, st[m]["Q"])
        if crb_w[m] is not None:
            rows, Jinv = crb_w[m]
            P = Jinv[:, :, :2]
            Gam = -np.einsum("kia,kja->kij", P, P)
            w = np.zeros((rows.size // 7, 7), complex)
            w[:, 0], w[:, 1], w[:, 2], w[:, 3] = Gam[:, 0, 0], Gam[:, 0, 1], Gam[:, 1, 0], Gam[:, 1, 1]
            w[:, 4] = 2 * Gam[:, 0, 2] + 2j * Gam[:, 0, 3]
            w[:, 5] = 2 * Gam[:, 1, 2] + 2j * Gam[:, 1, 3]
            w[:, 6] = Gam[:, 2, 2] + Gam[:, 3, 3]
            gc = (w.ravel() @ st[m]["Lc"][rows]).reshape(n, n).T
            g = g + (1 - eps) * gc
        g_common.append((g + g.conj().T) / 2)
    gW = {}
    for (m, k) in W:
        gw = g_common[m] - ck[k] * st[m]["Q"][k]
        gW[(m, k)] = (gw + gw.conj().T) / 2
    return f, (gW, g_common)


def _crb_blocks(theta, d, beta, U, nr):
    """Reduced matrices so that the FIM of Eqs. (16)-(18) is linear in X (Rx = Pt U X U^H).

    Order [d, theta, Re beta, Im beta], as in crb.fim_paper.
    """
    AH, dth, dd = crb._ah_with_derivs(theta, d, nr)
    A = AH.conj().T
    AHU = AH @ U
    BU = A @ AHU
    P = [dd.conj().T @ AHU + A @ (dd @ U), dth.conj().T @ AHU + A @ (dth @ U)]   # dB_a U
    c = 2 * 1 * config.PT_W / config.SIGMA2_R_W          # 2T Pt / sigma_z^2, T = 1 (gap 20)
    return dict(M=[[c * abs(beta) ** 2 * P[a].conj().T @ P[b] for b in range(2)] for a in range(2)],
                C=[c * np.conj(beta) * P[a].conj().T @ BU for a in range(2)],
                G=c * BU.conj().T @ BU)


def fim_from_x(blocks, X):
    """4 x 4 numeric FIM for a reduced covariance X (numpy)."""
    J = np.zeros((4, 4))
    for a in range(2):
        for b in range(2):
            J[a, b] = np.real(np.trace(blocks["M"][a][b] @ X))
        t = np.trace(blocks["C"][a] @ X)
        J[a, 2] = J[2, a] = np.real(t)
        J[a, 3] = J[3, a] = np.real(1j * t)
    J[2, 2] = J[3, 3] = np.real(np.trace(blocks["G"] @ X))
    return J


def crb_from_fim(J):
    """(CRB(d) m^2, CRB(theta) rad^2) via the Schur complement, Eq. (19)."""
    J11, J12, J22 = J[:2, :2], J[:2, 2:], J[2:, 2:]
    jbar = J11 - J12 @ np.linalg.solve(J22, J12.T)
    c = np.linalg.inv(jbar)
    return c[0, 0], c[1, 1]


def build_slot(theta, d, beta, nr=config.NR, tol=config.SUBSPACE_TOL):
    """Precompute the subspace, reduced channels and CRB blocks for one slot.

    theta, d, beta: (M, K) as seen from each RSU.
    """
    M, K = theta.shape
    gain = np.sqrt(config.PT_W / config.SIGMA2_C_W)
    U_all, Hs_all, fim_all, sc_all = [], [], [], []
    for m in range(M):
        cols, chans = [], []
        for k in range(K):
            AH, dth, dd = crb._ah_with_derivs(theta[m, k], d[m, k], nr)
            G = channel.channel_matrix(theta[m, k], d[m, k], nr=nr)       # Gamma (.) A^H, (Nr, Nt)
            chans.append(G.conj().T)                                       # H_{m,k}: (Nt, Nr)
            for blk in (AH.conj().T, dth.conj().T, dd.conj().T, G.conj().T):
                cols.append(blk / np.linalg.norm(blk, axis=0, keepdims=True))
        u, s, _ = np.linalg.svd(np.hstack(cols), full_matrices=False)
        U = u[:, s > tol * s[0]]
        U_all.append(U)
        Hs_all.append([gain * U.conj().T @ H for H in chans])
        blocks = [_crb_blocks(theta[m, k], d[m, k], beta[m, k], U, nr) for k in range(K)]
        fim_all.append(blocks)
        # LMI scaling (CVXPY path): FIM diagonal at an isotropic covariance in the subspace.
        x_ref = np.eye(U.shape[1]) / U.shape[1]
        sc_all.append([1.0 / np.sqrt(np.maximum(np.diag(fim_from_x(b, x_ref)), 1e-300)) for b in blocks])
    return SlotData(theta, d, beta, nr, U_all, Hs_all, fim_all, sc_all)


# ----------------------------------------------------------- exact evaluation
def interference(slot, xi, W, R, k):
    """Sum over all RSUs of interference covariance at vehicle k (Nr x Nr, noise units)."""
    M = xi.shape[0]
    out = np.zeros((slot.nr, slot.nr), complex)
    for m in range(M):
        H = slot.Hs[m][k]
        for kk in np.flatnonzero(xi[m]):
            if kk != k:
                out += H.conj().T @ W[(m, kk)] @ H
        out += H.conj().T @ R[m] @ H
    return out


def rates(slot, xi, W, R, rho, iota):
    """Semantic rate of Eq. (13) for every vehicle, bit/s/Hz. rho: (M, K)."""
    K = xi.shape[1]
    out = np.zeros(K)
    for k in range(K):
        m = int(np.argmax(xi[:, k]))
        H = slot.Hs[m][k]
        I = np.eye(slot.nr)
        Int = interference(slot, xi, W, R, k)
        Sig = H.conj().T @ W[(m, k)] @ H
        ld1 = np.linalg.slogdet(I + Sig + Int)[1]
        ld0 = np.linalg.slogdet(I + Int)[1]
        out[k] = iota / rho[m, k] * (ld1 - ld0) / LN2
    return out


def crbs(slot, xi, W, R):
    """(CRB(d), CRB(theta)) of every served pair, arrays (K,) for the serving RSU."""
    K = xi.shape[1]
    cd, ct = np.full(K, np.inf), np.full(K, np.inf)
    for k in range(K):
        m = int(np.argmax(xi[:, k]))
        X = R[m] + sum(W[(m, kk)] for kk in np.flatnonzero(xi[m]))
        try:
            cd[k], ct[k] = crb_from_fim(fim_from_x(slot.fim[m][k], X))
        except np.linalg.LinAlgError:
            pass
    return cd, ct


def beam_power(xi, W, R):
    """P_C&S of Eq. (21) per RSU, in W."""
    M = xi.shape[0]
    return np.array([config.PT_W * np.real(np.trace(R[m]) + sum(np.trace(W[(m, k)]) for k in np.flatnonzero(xi[m])))
                     for m in range(M)])


def objective(slot, xi, W, R, rho, iota, eps=config.EPS_WEIGHT):
    """Value of (41a) with the exact rate (nats) and exact CRB trace (rad^2 + m^2)."""
    s = rates(slot, xi, W, R, rho, iota) * LN2
    cd, ct = crbs(slot, xi, W, R)
    return float(np.sum(-eps * s + (1 - eps) * (cd + ct)))


def surrogate_reference(slot, xi, W, R, rho, iota, eps=config.EPS_WEIGHT, grad=False):
    """Problem (41)'s objective with eta_S and eta at their optimal values.

    f = sum_{served k} -eps (iota / rho_k) (ln|E_k| - Tr(Int_k)) + (1 - eps) (CRB_d + CRB_theta)
    E_k = I + Sig_k + Int_k; the interference uses Lemma 2 (Eq. 38) as in (39).
    Equal to (41a) at the optimum, because eta_S and eta are epigraph variables.
    With grad=True also returns the Hermitian gradients, as (gW dict, gR list).
    Returns +inf where a CRB does not exist (a served vehicle gets no energy).
    """
    M, K = xi.shape
    T = [R[m] + sum((W[(m, k)] for k in np.flatnonzero(xi[m])), np.zeros_like(R[m])) for m in range(M)]
    g_common = [np.zeros(R[m].shape, complex) for m in range(M)]
    g_w = {key: np.zeros(v.shape, complex) for key, v in W.items()}
    f = 0.0
    I = np.eye(slot.nr)
    for k in range(K):
        m = int(np.argmax(xi[:, k]))
        ck = eps * iota / rho[m, k]
        E = I.astype(complex)
        tr_int = 0.0
        for mm in range(M):
            H = slot.Hs[mm][k]
            E = E + H.conj().T @ T[mm] @ H
            tr_int += np.real(np.trace(H.conj().T @ T[mm] @ H))
        H = slot.Hs[m][k]
        tr_int -= np.real(np.trace(H.conj().T @ W[(m, k)] @ H))
        sign, ld = np.linalg.slogdet(E)
        f += -ck * (ld - tr_int)
        # CRB term at the serving RSU
        J = fim_from_x(slot.fim[m][k], T[m])
        try:
            Jinv = np.linalg.inv(J)
            if not np.all(np.isfinite(Jinv)) or Jinv[0, 0] <= 0 or Jinv[1, 1] <= 0:
                return (np.inf, None) if grad else np.inf
        except np.linalg.LinAlgError:
            return (np.inf, None) if grad else np.inf
        f += (1 - eps) * (Jinv[0, 0] + Jinv[1, 1])
        if not grad:
            continue
        Einv = np.linalg.inv(E)
        for mm in range(M):
            Hm = slot.Hs[mm][k]
            g_common[mm] += -ck * (Hm @ Einv @ Hm.conj().T - Hm @ Hm.conj().T)
        g_w[(m, k)] += -ck * (H @ H.conj().T)
        # d Tr(P^T J^-1 P) / dJ = -J^-1 P P^T J^-1, with P picking [d, theta]
        Pm = Jinv[:, :2]
        Gam = -(Pm @ Pm.T)
        b = slot.fim[m][k]
        gc = np.zeros(T[m].shape, complex)
        for a in range(2):
            for c in range(2):
                gc += Gam[a, c] * b["M"][a][c]
            gc += 2 * Gam[a, 2] * b["C"][a] + 2 * Gam[a, 3] * 1j * b["C"][a]
        gc += (Gam[2, 2] + Gam[3, 3]) * b["G"]
        g_common[m] += (1 - eps) * (gc + gc.conj().T) / 2
    if not grad:
        return f
    gW = {key: g_common[key[0]] + g_w[key] for key in W}
    gW = {key: (v + v.conj().T) / 2 for key, v in gW.items()}
    gR = [(g + g.conj().T) / 2 for g in g_common]
    return f, (gW, gR)


def _project(W, R, xi, budget):
    """Projection onto {all matrices PSD, sum of traces per RSU <= budget[m]} (Frobenius norm)."""
    M = xi.shape[0]
    Wp, Rp = {}, list(R)
    for m in range(M):
        keys = [(m, k) for k in np.flatnonzero(xi[m])]
        mats = [W[key] for key in keys] + [R[m]]
        eig = [np.linalg.eigh((X + X.conj().T) / 2) for X in mats]
        lam = np.concatenate([w for w, _ in eig])
        mu = _project_capped(lam, budget[m])
        pos = 0
        out = []
        for w, v in eig:
            mw = mu[pos:pos + w.size]
            pos += w.size
            out.append((v * mw) @ v.conj().T)
        for key, X in zip(keys, out[:-1]):
            Wp[key] = X
        Rp[m] = out[-1]
    return Wp, Rp


def _project_capped(lam, cap):
    """Euclidean projection of a vector onto {mu >= 0, sum(mu) <= cap}."""
    mu = np.maximum(lam, 0)
    if mu.sum() <= cap:
        return mu
    s = np.sort(lam)[::-1]
    css = np.cumsum(s)
    idx = np.arange(1, s.size + 1)
    r = np.nonzero(s - (css - cap) / idx > 0)[0][-1]
    tau = (css[r] - cap) / (r + 1)
    return np.maximum(lam - tau, 0)


def initial_beams(slot, xi, budget):
    """Feasible start: matched beams to served vehicles plus an isotropic R, equal power shares."""
    M = xi.shape[0]
    W, R = {}, []
    for m in range(M):
        served = np.flatnonzero(xi[m])
        n = slot.U[m].shape[1]
        share = budget[m] / (served.size + 1)
        for k in served:
            u = np.linalg.svd(slot.Hs[m][k], full_matrices=False)[0][:, 0]
            W[(m, k)] = share * np.outer(u, u.conj())
        R.append(share * np.eye(n, dtype=complex) / n)
    return W, R


def solve_beams_spg(slot, xi, rho, p_avail, iota, eps=config.EPS_WEIGHT, start=None,
                    max_iter=3000, rtol=1e-9, memory=10, trace=None):
    """Alg. 3 line 3 by spectral projected gradient (Birgin, Martinez and Raydan, 2000).

    Same convex problem as solve_beams_cvx (eta_S, eta eliminated), solved directly on the
    PSD matrices. Nonmonotone Armijo line search; Barzilai-Borwein steps.
    """
    M, K = xi.shape
    if np.any(p_avail <= 0):
        return None
    budget = p_avail / config.PT_W
    W, R = _project(*(start or initial_beams(slot, xi, budget)), xi, budget)
    f, (gW, gR) = surrogate(slot, xi, W, R, rho, iota, eps, grad=True)
    if not np.isfinite(f):
        return None
    keys = list(W)

    def flat(Wd, Rl):
        return np.concatenate([Wd[k].ravel() for k in keys] + [r.ravel() for r in Rl])

    def inner(a, b):
        return float(np.real(np.vdot(a, b)))

    hist = [f]
    step = 1.0 / max(np.abs(flat(gW, gR)).max(), 1e-12)
    for it in range(max_iter):
        Wt = {k: W[k] - step * gW[k] for k in keys}
        Rt = [R[m] - step * gR[m] for m in range(M)]
        Wp, Rp = _project(Wt, Rt, xi, budget)
        dW = {k: Wp[k] - W[k] for k in keys}
        dR = [Rp[m] - R[m] for m in range(M)]
        dvec = flat(dW, dR)
        gvec = flat(gW, gR)
        slope = inner(gvec, dvec)
        if slope >= 0 or np.linalg.norm(dvec) < 1e-14:
            break
        fmax, lam = max(hist[-memory:]), 1.0
        while True:
            Wn = {k: W[k] + lam * dW[k] for k in keys}
            Rn = [R[m] + lam * dR[m] for m in range(M)]
            fn = surrogate(slot, xi, Wn, Rn, rho, iota, eps)
            if np.isfinite(fn) and fn <= fmax + 1e-4 * lam * slope:
                break
            lam *= 0.5
            if lam < 1e-12:
                break
        if lam < 1e-12:
            break
        fn, (gWn, gRn) = surrogate(slot, xi, Wn, Rn, rho, iota, eps, grad=True)
        s = flat({k: Wn[k] - W[k] for k in keys}, [Rn[m] - R[m] for m in range(M)])
        y = flat(gWn, gRn) - gvec
        sy = inner(s, y)
        step = inner(s, s) / sy if sy > 0 else step * 2
        rel = abs(f - fn) / max(abs(f), 1e-12)
        W, R, f, gW, gR = Wn, Rn, fn, gWn, gRn
        hist.append(f)
        if trace is not None:
            trace.append(f)
        if rel < rtol and it > 5:
            break
    eta_s = np.zeros(K)
    for k in range(K):
        m = int(np.argmax(xi[:, k]))
        Int = interference(slot, xi, W, R, k)
        E = np.eye(slot.nr) + slot.Hs[m][k].conj().T @ W[(m, k)] @ slot.Hs[m][k] + Int
        eta_s[k] = iota / rho[m, k] * (np.linalg.slogdet(E)[1] - np.real(np.trace(Int)))
    return dict(W=W, R=R, eta_S=eta_s, status=f"spg_{it + 1}_iters", value=f)


def _eta_s(slot, xi, W, R, rho, iota):
    """eta_S of every vehicle at the optimum of (39): (iota / rho)(ln|E| - Tr(Int)), nats."""
    K = xi.shape[1]
    out = np.zeros(K)
    for k in range(K):
        m = int(np.argmax(xi[:, k]))
        Int = interference(slot, xi, W, R, k)
        E = np.eye(slot.nr) + slot.Hs[m][k].conj().T @ W[(m, k)] @ slot.Hs[m][k] + Int
        out[k] = iota / rho[m, k] * (np.linalg.slogdet(E)[1] - np.real(np.trace(Int)))
    return out


def full_rank_start(slot, xi, budget, start=None, mix=1e-3):
    """Starting point for MEG: given beams (or matched beams) mixed with a little isotropic power,
    rescaled so every RSU uses its whole budget."""
    W, R = start if start is not None else initial_beams(slot, xi, budget)
    W, R = dict(W), list(R)
    M = xi.shape[0]
    for m in range(M):
        served = np.flatnonzero(xi[m])
        n = slot.U[m].shape[1]
        for k in served:
            if (m, k) not in W:                      # vehicle new to this RSU (SA move)
                u = np.linalg.svd(slot.Hs[m][k], full_matrices=False)[0][:, 0]
                W[(m, k)] = budget[m] / (served.size + 1) * np.outer(u, u.conj())
        mats = [W[(m, k)] for k in served] + [R[m]]
        tot = sum(np.real(np.trace(X)) for X in mats)
        iso = mix * budget[m] / len(mats) * np.eye(n) / n
        scale = (1 - mix) * budget[m] / tot if tot > 0 else 0.0
        for k in served:
            W[(m, k)] = scale * W[(m, k)] + iso
        R[m] = scale * R[m] + iso
    W = {key: v for key, v in W.items() if xi[key[0], key[1]]}
    return W, R


def solve_beams_meg(slot, xi, rho, p_avail, iota, eps=config.EPS_WEIGHT, start=None,
                    max_iter=config.MEG_MAX_ITER, rtol=1e-8, floor=1e-14, trace=None):
    """Alg. 3 line 3 by matrix exponentiated gradient with a backtracking line search.

    The update X <- exp(log X - t G), rescaled so each RSU's matrices use the whole budget,
    is a mirror-descent step on the PSD cone (Tsuda, Ratsch and Warmuth, JMLR 2005). It moves
    small eigenvalues multiplicatively, which suits the CRB terms (which behave like 1/energy).
    Every RSU spends its full budget: more power never raises (41a).
    """
    M, K = xi.shape
    if np.any(p_avail <= 0):
        return None
    budget = p_avail / config.PT_W
    W, R = full_rank_start(slot, xi, budget, start)
    f, (gW, gR) = surrogate(slot, xi, W, R, rho, iota, eps, grad=True)
    if not np.isfinite(f):
        return None
    gmax = max(max((np.abs(g).max() for g in gW.values()), default=0), max(np.abs(g).max() for g in gR))
    t = 1.0 / max(gmax, 1e-12)
    it = 0
    for it in range(max_iter):
        logs = {key: _logm(X, floor) for key, X in W.items()}
        logs_r = [_logm(X, floor) for X in R]
        while True:
            Wn, Rn = {}, []
            for m in range(M):
                keys = [key for key in W if key[0] == m]
                eig = [np.linalg.eigh(logs[key] - t * gW[key]) for key in keys]
                eig.append(np.linalg.eigh(logs_r[m] - t * gR[m]))
                top = max(w.max() for w, _ in eig)
                tot = sum(np.exp(w - top).sum() for w, _ in eig)
                mats = [(v * (np.exp(w - top) * budget[m] / tot)) @ v.conj().T for w, v in eig]
                for key, X in zip(keys, mats[:-1]):
                    Wn[key] = X
                Rn.append(mats[-1])
            fn = surrogate(slot, xi, Wn, Rn, rho, iota, eps)
            if np.isfinite(fn) and fn <= f:
                break
            t *= 0.5
            if t * gmax < 1e-14:
                break
        if not (np.isfinite(fn) and fn <= f):
            break
        fn, (gWn, gRn) = surrogate(slot, xi, Wn, Rn, rho, iota, eps, grad=True)
        rel = abs(f - fn) / max(abs(f), 1e-12)
        W, R, f, gW, gR = Wn, Rn, fn, gWn, gRn
        t *= 1.5
        if trace is not None:
            trace.append(f)
        if rel < rtol:
            break
    return dict(W=W, R=R, eta_S=_eta_s(slot, xi, W, R, rho, iota), status=f"meg_{it + 1}_iters", value=f)


def _logm(X, floor):
    w, v = np.linalg.eigh((X + X.conj().T) / 2)
    return (v * np.log(np.maximum(w, floor))) @ v.conj().T


def solve_beams_lbfgs(slot, xi, rho, p_avail, iota, eps=config.EPS_WEIGHT, start=None,
                      max_iter=config.LBFGS_MAX_ITER, trace=None, extra=None):
    """Alg. 3 line 3 by L-BFGS on a factorised form (our default solver).

    Each matrix is written X = (P_m / S_m) Y Y^H with S_m the sum of ||Y||_F^2 over the
    matrices of RSU m, so X is PSD and the RSU spends exactly its budget P_m for any Y.
    This is the Burer-Monteiro idea (Burer and Monteiro, Math. Program., 2003): with Y
    square (full rank allowed), a local minimum where Y is rank-deficient is a global
    minimum of the convex problem (41). Checked against CLARABEL in the tests.
    More power never raises (41a), so using the whole budget loses nothing.
    extra(W, R) -> (f, gW, gR) adds a term to the objective (used by baselines.py for [18]).
    """
    from scipy.optimize import minimize

    M, K = xi.shape
    if np.any(p_avail <= 0):
        return None
    budget = p_avail / config.PT_W
    W0, R0 = full_rank_start(slot, xi, budget, start, mix=1e-2)
    blocks = [(m, key) for m in range(M) for key in list(W0) if key[0] == m] + [(m, None) for m in range(M)]
    blocks.sort(key=lambda b: b[0])

    def sqrt_psd(X):
        w, v = np.linalg.eigh((X + X.conj().T) / 2)
        return v * np.sqrt(np.maximum(w, 0))

    Y0 = [sqrt_psd(R0[m] if key is None else W0[key]) for m, key in blocks]
    shapes = [y.shape for y in Y0]

    def pack(ys):
        return np.concatenate([np.concatenate([y.real.ravel(), y.imag.ravel()]) for y in ys])

    def unpack(z):
        ys, p = [], 0
        for sh in shapes:
            n = sh[0] * sh[1]
            ys.append(z[p:p + n].reshape(sh) + 1j * z[p + n:p + 2 * n].reshape(sh))
            p += 2 * n
        return ys

    def to_mats(ys):
        S = np.zeros(M)
        for (m, _), y in zip(blocks, ys):
            S[m] += np.sum(np.abs(y) ** 2)
        Wd, Rl, xs = {}, [None] * M, []
        for (m, key), y in zip(blocks, ys):
            X = budget[m] / S[m] * (y @ y.conj().T)
            xs.append(X)
            if key is None:
                Rl[m] = X
            else:
                Wd[key] = X
        return Wd, Rl, xs, S

    @np.errstate(all="ignore")
    def fg(z):
        ys = unpack(z)
        Wd, Rl, xs, S = to_mats(ys)
        if np.any(S <= 0):
            return 1e30, np.zeros_like(z)
        f, g = surrogate(slot, xi, Wd, Rl, rho, iota, eps, grad=True)
        if not np.isfinite(f):
            return 1e30, np.zeros_like(z)
        gW, gR = g
        if extra is not None:
            fe, eW, eR = extra(Wd, Rl)
            f += fe
            gW = {key: gW[key] + eW[key] for key in gW}
            gR = [gR[m] + eR[m] for m in range(M)]
        Gs = [gR[m] if key is None else gW[key] for m, key in blocks]
        inner = np.zeros(M)
        for (m, _), G, X in zip(blocks, Gs, xs):
            inner[m] += np.real(np.trace(G @ X))
        grads = [2 * budget[m] / S[m] * (G @ y - inner[m] / budget[m] * y)
                 for (m, _), G, y in zip(blocks, Gs, ys)]
        if trace is not None:
            trace.append(f)
        return f, pack(grads)

    if not np.isfinite(surrogate(slot, xi, W0, R0, rho, iota, eps)):
        return None
    res = minimize(fg, pack(Y0), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=max_iter, maxcor=10, ftol=1e-12, gtol=1e-10))
    W, R, _, _ = to_mats(unpack(res.x))
    W = {key: _psd(v) for key, v in W.items()}
    R = [_psd(v) for v in R]
    return dict(W=W, R=R, eta_S=_eta_s(slot, xi, W, R, rho, iota), status=f"lbfgs_{res.nit}_iters",
                value=surrogate(slot, xi, W, R, rho, iota, eps))


def solve_beams(slot, xi, rho, p_avail, iota, eps=config.EPS_WEIGHT, solver="LBFGS", start=None, max_iter=None):
    """Alg. 3 line 3 with the chosen solver ("LBFGS", "MEG", "SPG", or a CVXPY solver name)."""
    kw = {} if max_iter is None else dict(max_iter=max_iter)
    if solver == "LBFGS":
        return solve_beams_lbfgs(slot, xi, rho, p_avail, iota, eps, start=start, **kw)
    if solver == "MEG":
        kw = {} if max_iter is None else dict(max_iter=max_iter)
        return solve_beams_meg(slot, xi, rho, p_avail, iota, eps, start=start, **kw)
    if solver == "SPG":
        return solve_beams_spg(slot, xi, rho, p_avail, iota, eps, start=start)
    return solve_beams_cvx(slot, xi, rho, p_avail, iota, eps, solver=solver)


# ------------------------------------------------------------- convex step (CVXPY)
def solve_beams_cvx(slot, xi, rho, p_avail, iota, eps=config.EPS_WEIGHT, solver="CLARABEL"):
    """Alg. 3 line 3: problem (41) for W, R, eta_S, eta, Omega with rho fixed.

    p_avail: (M,) power left for beams after P_Comp and P_DT, in W.
    Returns dict(W, R, eta_S (K,), status, value) or None if infeasible.
    """
    M, K = xi.shape
    if np.any(p_avail <= 0):
        return None
    W = {(m, k): cp.Variable((slot.U[m].shape[1],) * 2, hermitian=True)
         for m in range(M) for k in np.flatnonzero(xi[m])}
    R = [cp.Variable((slot.U[m].shape[1],) * 2, hermitian=True) for m in range(M)]
    eta_s = cp.Variable(K)
    cons = [X >> 0 for X in W.values()] + [X >> 0 for X in R]
    obj = 0
    for m in range(M):
        served = np.flatnonzero(xi[m])
        tot = R[m] + sum(W[(m, k)] for k in served)
        cons.append(cp.real(cp.trace(tot)) <= p_avail[m] / config.PT_W)     # (34e)
        for k in served:
            # Rate bound (39) with Lemma 2 for the interference and exact -ln|E| (deviation 1).
            int_tr = 0
            E = np.eye(slot.nr) + slot.Hs[m][k].conj().T @ W[(m, k)] @ slot.Hs[m][k]
            for mm in range(M):
                H = slot.Hs[mm][k]
                Q = H @ H.conj().T
                for kk in np.flatnonzero(xi[mm]):
                    if kk != k:
                        int_tr += cp.real(cp.trace(Q @ W[(mm, kk)]))
                        E = E + H.conj().T @ W[(mm, kk)] @ H
                int_tr += cp.real(cp.trace(Q @ R[mm]))
                E = E + H.conj().T @ R[mm] @ H
            cons.append(iota / rho[m, k] * (int_tr - cp.log_det(E)) <= -eta_s[k])
            # CRB LMI (40) in scaled form: J~ = D J D, Omega~ = D1 Omega D1.
            b, D = slot.fim[m][k], slot.scale[m][k]
            J = [[None] * 4 for _ in range(4)]
            for a in range(2):
                for c in range(a, 2):
                    J[a][c] = J[c][a] = D[a] * D[c] * cp.real(cp.trace(b["M"][a][c] @ tot))
                t = b["C"][a]
                J[a][2] = J[2][a] = D[a] * D[2] * cp.real(cp.trace(t @ tot))
                J[a][3] = J[3][a] = D[a] * D[3] * cp.real(cp.trace(1j * t @ tot))
            J[2][2] = D[2] ** 2 * cp.real(cp.trace(b["G"] @ tot))
            J[3][3] = D[3] ** 2 * cp.real(cp.trace(b["G"] @ tot))
            J[2][3] = J[3][2] = cp.Constant(0.0)
            om = cp.Variable((2, 2), symmetric=True)
            lmi = cp.bmat([[J[0][0] - om[0, 0], J[0][1] - om[0, 1], J[0][2], J[0][3]],
                           [J[1][0] - om[1, 0], J[1][1] - om[1, 1], J[1][2], J[1][3]],
                           [J[2][0], J[2][1], J[2][2], J[2][3]],
                           [J[3][0], J[3][1], J[3][2], J[3][3]]])
            cons += [(lmi + lmi.T) / 2 >> 0, om >> 0]
            # eta >= Tr(Omega^-1) in SI: Omega^-1 = D1 Omega~^-1 D1 -> Tr = matrix_frac(D1, Omega~).
            eta = cp.matrix_frac(np.diag(D[:2]), om)
            obj += -eps * eta_s[k] + (1 - eps) * eta
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=solver)
    except cp.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    Wv = {key: _psd(v.value) for key, v in W.items()}
    Rv = [_psd(v.value) for v in R]
    return dict(W=Wv, R=Rv, eta_S=np.array(eta_s.value), status=prob.status, value=prob.value)


def _psd(X):
    """Clean a solver output: Hermitian part with negative eigenvalues set to 0."""
    X = (X + X.conj().T) / 2
    w, v = np.linalg.eigh(X)
    return (v * np.maximum(w, 0)) @ v.conj().T


# ------------------------------------------------------------- rho step
def rho_step(slot, xi, W, R, rho, p_left, iota, eps=config.EPS_WEIGHT, tol=1e-4):
    """Alg. 3 line 5: bisection for one rho per RSU (deviation 5).

    For a trial rho the beams of that RSU are rescaled so P_C&S = p_left - P_Comp(rho).
    p_left: (M,) power after P_DT, in W. Returns the new rho (M, K).
    """
    M, K = xi.shape
    rho = rho.copy()
    for m in range(M):
        served = np.flatnonzero(xi[m])
        if served.size == 0:
            continue
        used = beam_power(xi, W, R)[m]

        def value(r):
            avail = p_left[m] - allocation.comp_power(np.full(served.size, r))
            if avail <= 0 or used <= 0:
                return -np.inf
            s = avail / used
            Ws = {key: (v * s if key[0] == m else v) for key, v in W.items()}
            Rs = [R[j] * s if j == m else R[j] for j in range(M)]
            rr = rho.copy()
            rr[m, served] = r
            return -surrogate(slot, xi, Ws, Rs, rr, iota, eps)

        lo, hi = config.RHO_LB, 1.0
        h = 1e-5
        slope = lambda r: value(min(r + h, 1.0)) - value(max(r - h, config.RHO_LB))
        if slope(lo + h) <= 0:          # objective falls as rho grows: best at the lower bound
            best = lo
        elif slope(hi - h) >= 0:
            best = hi
        else:
            while hi - lo > tol:
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if slope(mid) > 0 else (lo, mid)
            best = (lo + hi) / 2
        rho[m, served] = best
    return rho


# ------------------------------------------------------------- Alg. 3
def randomise(slot, xi, W, R, rho, iota, rng, n=config.N_RANDOMISATION, eps=config.EPS_WEIGHT):
    """Alg. 3 line 8: Gaussian randomisation of each W to rank one, keeping its power."""
    best, best_val = None, np.inf
    for _ in range(n):
        Wr = {}
        for key, X in W.items():
            w, v = np.linalg.eigh(X)
            z = (rng.standard_normal(w.size) + 1j * rng.standard_normal(w.size)) / np.sqrt(2)
            vec = v @ (np.sqrt(np.maximum(w, 0)) * z)
            nrm = np.linalg.norm(vec)
            vec = vec * np.sqrt(np.real(np.trace(X))) / nrm if nrm > 0 else vec
            Wr[key] = np.outer(vec, vec.conj())
        val = objective(slot, xi, Wr, R, rho, iota, eps)
        if val < best_val:
            best, best_val = Wr, val
    return best, best_val


def _plan(xi, c, dbits, nu3, cd, ct, latency):
    """CPU plan for the current CRBs. latency="as_written": L of Eq. (24) from the RCRBs;
    "relaxed": L = |nu3| only (sensing error left out of the DT workload; sensitivity run)."""
    if latency == "as_written":
        l_extra = allocation.workload(np.rad2deg(np.sqrt(ct)), np.sqrt(cd), nu3)
    elif latency == "relaxed":
        l_extra = np.abs(nu3)
    else:
        raise ValueError(latency)
    return allocation.computing_plan(xi, c, dbits, np.where(np.isfinite(l_extra), l_extra, 1e30))


def inner_problem(slot, xi, c, dbits, nu3, iota=config.IOTA, semantic=True, rng=None,
                  eps=config.EPS_WEIGHT, randomisation=True, solver="LBFGS", max_iter=None,
                  start=None, latency="as_written", info=None, rho0=None, ao_max=config.AO_MAX_ITERS):
    """Alg. 3 for one assignment xi (M, K). Returns a result dict, or None if infeasible.

    semantic=False is the "non-semantic HH" (gap 31): rho = 1, iota = 1, no P_Comp.
    L in Eq. (24) uses the RCRBs of the "previous slot". We take the previous AO iteration
    as that slot (same geometry; cars move 0.4 m per slot), starting from L = |nu3|.
    If (34g)/(34h) fail at the end, the assignment is infeasible (info["reason"] says why).
    """
    M, K = xi.shape
    info = {} if info is None else info
    if not semantic:
        iota = 1.0
    rho = np.full((M, K), config.RHO_0 if semantic else 1.0)
    if semantic and rho0 is not None:
        rho[:] = rho0                                      # warm start (used inside SA)
    cd = ct = np.zeros(K)                                  # no sensing error known yet
    prev, res, history = None, None, []
    for it in range(ao_max):
        f, p_dt, ok = _plan(xi, c, dbits, nu3, cd, ct, latency)
        if not np.all(ok):                                 # (34g)/(34h)
            info["reason"] = "cpu"
            return None
        p_left = config.PT_W - p_dt
        p_comp = np.array([allocation.comp_power(rho[m, xi[m] > 0]) if semantic else 0.0 for m in range(M)])
        short = p_left - p_comp <= 0
        if semantic and np.any(short & (p_left > 0)):      # a warm-started rho is too costly here:
            rho[short] = 1.0                               # rho = 1 needs no P_Comp, so it is always allowed
            p_comp[short] = 0.0
        if np.any(p_left - p_comp <= 0):                   # (34e) cannot leave power for beams
            info["reason"] = "power"
            return None
        warm = start if res is None else (res["W"], res["R"])
        res = solve_beams(slot, xi, rho, p_left - p_comp, iota, eps, solver=solver, start=warm, max_iter=max_iter)
        if res is None:
            info["reason"] = "solver"
            return None
        if semantic:
            rho = rho_step(slot, xi, res["W"], res["R"], rho, p_left, iota, eps)
        cd, ct = crbs(slot, xi, res["W"], res["R"])
        total = float(np.sum(res["eta_S"]))
        history.append(total)
        if prev is not None and abs(total - prev) < config.EPS_TOL:    # Alg. 3 line 7
            break
        prev = total
    f, p_dt, ok = _plan(xi, c, dbits, nu3, cd, ct, latency)
    if not np.all(ok):
        info["reason"] = "cpu"
        return None
    p_left = config.PT_W - p_dt
    # The last rho step may have changed rho: rescale the beams to the power left.
    W, R = res["W"], res["R"]
    p_comp = np.array([allocation.comp_power(rho[m, xi[m] > 0]) if semantic else 0.0 for m in range(M)])
    for m in range(M):
        used = beam_power(xi, W, R)[m]
        avail = p_left[m] - p_comp[m]
        if avail <= 0:
            info["reason"] = "power"
            return None
        if used > avail:
            s = avail / used
            W = {key: (v * s if key[0] == m else v) for key, v in W.items()}
            R = [R[j] * s if j == m else R[j] for j in range(M)]
    value = surrogate(slot, xi, W, R, rho, iota, eps)
    relaxed = dict(W=W, R=R)
    if randomisation:
        W, _ = randomise(slot, xi, W, R, rho, iota, rng or np.random.default_rng(0), eps=eps)
    s = rates(slot, xi, W, R, rho, iota)
    cd, ct = crbs(slot, xi, W, R)
    return dict(score=-value, value_41a=value, W=W, R=R, relaxed=relaxed, rho=rho,
                rate_bps=s, crb_d=cd, crb_theta=ct, f=f, p_dt=p_dt, p_cs=beam_power(xi, W, R),
                p_comp=p_comp, iterations=len(history), history=history)
