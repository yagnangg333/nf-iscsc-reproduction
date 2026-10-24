"""Cramer-Rao bounds (Stage 4).

1) The paper's CRB, Sec. III-B, Eqs. (15)-(19), for Psi = [d, theta, beta]:
     J_ab      = 2T|beta|^2 / sigma_z^2 * Re Tr(dB_b Rx dB_a^H)              (16)
     J_a,beta  = 2T / sigma_z^2 * Re{ beta^* Tr(B Rx dB_a^H) [1, j] }      (17)
     J_bb      = 2T / sigma_z^2 * Tr(B Rx B^H) I_2                         (18)
     CRB(d) = Jbar^-1[1,1], CRB(theta) = Jbar^-1[2,2],
     Jbar = J11 - J12 J22^-1 J12^T  (Schur complement)                     (19)
   with B = A(theta, d) A^H(theta, d) (Eq. 5) and Rx = W W^H + R R^H (Eq. 3).
   Deviations (docs/stage4-notes.md): Re{} added in (16) (gap 19), beta^* moved inside
   Re{} in (17), and B uses d where the paper prints s_{m,k,i}.
   Rx is passed as a beam matrix X (Nt x L) with Rx = X X^H.

2) A posterior (Bayesian) CRB for the Stage 3 filters (not in the paper; ours). A filter
   uses the motion model and many slots, so a one-snapshot CRB is not a lower bound for
   it. The right bound is the recursion of Tichavsky et al. (1998):
     J_{i+1} = D22 - D21 (J_i + D11)^-1 D12,
     D11 = E[F^T Q^-1 F], D12 = -E[F^T] Q^-1, D22 = Q^-1 + E[H^T R^-1 H],
   with F = dg1/dq, H = dg2/dq, Q from Eq. (12), R the stacked measurement noise.
"""
import numpy as np

import config
from tracking import models


# ----------------------------------------------------------- B and its derivatives
def _ah_with_derivs(theta, d, nr=config.NR):
    """A^H (Nr x Nt) and its derivatives w.r.t. theta and d, from Eq. (5)."""
    k = 2 * np.pi / config.WAVELENGTH
    psi, dpsi_th, dpsi_d = models._phase_parts(np.atleast_1d(theta), np.atleast_1d(d), nr)
    AH = np.exp(-1j * k * psi[0])
    return AH, -1j * k * dpsi_th[0] * AH, -1j * k * dpsi_d[0] * AH


def b_products(theta, d, X):
    """B X, (dB/dtheta) X and (dB/dd) X for a beam matrix X (Nt x L).

    B = A A^H with A = (A^H)^H, so dB = dA A^H + A dA^H, applied to X without
    forming the Nt x Nt matrix.
    """
    AH, dth, dd = _ah_with_derivs(theta, d)
    Y = AH @ X                                           # A^H X, (Nr, L)
    BX = AH.conj().T @ Y
    dBX_th = dth.conj().T @ Y + AH.conj().T @ (dth @ X)
    dBX_d = dd.conj().T @ Y + AH.conj().T @ (dd @ X)
    return BX, dBX_th, dBX_d


# ------------------------------------------------------------------- paper CRB
def fim_paper(theta, d, beta, X, sigma2=config.SIGMA2_R_W, T=1):
    """4 x 4 FIM of Eqs. (15)-(18) for [d, theta, Re(beta), Im(beta)]."""
    BX, dth, dd = b_products(theta, d, X)
    derivs = (dd, dth)                                  # order of Psi: d first, then theta
    c = 2 * T / sigma2
    J = np.zeros((4, 4))
    for a in range(2):
        for b in range(2):
            # Tr(dB_b Rx dB_a^H) = sum over beams of (dB_a x)^H (dB_b x)
            J[a, b] = c * abs(beta) ** 2 * np.real(np.vdot(derivs[a], derivs[b]))
        t = np.vdot(derivs[a], BX)                      # Tr(B Rx dB_a^H)
        J[a, 2] = J[2, a] = c * np.real(np.conj(beta) * t)
        J[a, 3] = J[3, a] = c * np.real(1j * np.conj(beta) * t)
    J[2, 2] = J[3, 3] = c * np.real(np.vdot(BX, BX))   # Tr(B Rx B^H)
    return J


def crb_paper(theta, d, beta, X, sigma2=config.SIGMA2_R_W, T=1):
    """(CRB(d) in m^2, CRB(theta) in rad^2) from Eq. (19)."""
    J = fim_paper(theta, d, beta, X, sigma2, T)
    J11, J12, J22 = J[:2, :2], J[:2, 2:], J[2:, 2:]
    jbar = J11 - J12 @ np.linalg.solve(J22, J12.T)
    c = np.linalg.inv(jbar)
    return c[0, 0], c[1, 1]


# ------------------------------------------------- bounds for the Stage 3 filters
def snapshot_crb(q, x, q2=config.Q2_DIAG):
    """CRB of one slot's stacked measurement (echo + d_hat + v_hat) alone, no prior.

    Diagonal of (H^T R^-1 H)^-1 for the state [theta, d, v, beta].
    """
    H = models.g2_jacobian(q, x)
    info = H.T @ (H / models.r_diag(q2)[:, None])
    return np.diag(np.linalg.inv(info))


def posterior_crb(tracks, beams, q1_si, cov0_diag, q2=config.Q2_DIAG, beta_noise=config.BETA_NOISE):
    """Posterior CRB per slot, averaged over Monte Carlo tracks of one vehicle.

    tracks: (M, n + 1, 4) true state sequences; beams: (M, n, Nt) probe beams x for slots 1..n.
    Returns (n, 4): diagonal of J_i^-1 for i = 1..n.
    With "relative" beta noise Q depends on beta; we evaluate Q at the true state and
    ignore its derivative (an approximation, stated in docs/stage4-notes.md).
    """
    M, n1, _ = tracks.shape
    r_inv = 1.0 / models.r_diag(q2)
    J = np.diag(1.0 / np.asarray(cov0_diag))
    out = np.zeros((n1 - 1, 4))
    for i in range(1, n1):
        D11 = np.zeros((4, 4))
        D12 = np.zeros((4, 4))
        D22 = np.zeros((4, 4))
        for mc in range(M):
            q_prev = tracks[mc, i - 1]
            Q_inv = np.diag(1.0 / models.process_noise_std(q_prev, q1_si, beta_noise) ** 2)
            F = models.g1_jacobian(q_prev)
            H = models.g2_jacobian(tracks[mc, i], beams[mc, i - 1])
            D11 += F.T @ Q_inv @ F
            D12 += -F.T @ Q_inv
            D22 += Q_inv + H.T @ (r_inv[:, None] * H)
        D11, D12, D22 = D11 / M, D12 / M, D22 / M
        J = D22 - D12.T @ np.linalg.solve(J + D11, D12)
        J = (J + J.T) / 2
        out[i - 1] = np.diag(np.linalg.inv(J))
    return out
