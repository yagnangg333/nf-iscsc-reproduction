"""Unscented Kalman filter on the same models as the PF and EKF.

The paper cites [55] for its UKF and does not give its settings. We use the
standard scaled unscented transform (Wan and van der Merwe) with
alpha = 1e-3, beta = 2, kappa = 0 (ASSUMED, gap 18).

Because R is diagonal and there are only 2n + 1 = 9 sigma points, the innovation
covariance S = R + D Wc D^T is inverted with the Woodbury identity
(a 9 x 9 solve instead of a (2 Nt + 2)-square one).
"""
import numpy as np

import config
from tracking import models

ALPHA, BETA_UT, KAPPA = 1e-3, 2.0, 0.0


def _weights(n):
    lam = ALPHA**2 * (n + KAPPA) - n
    wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + (1 - ALPHA**2 + BETA_UT)
    return lam, wm, wc


def _sigma_points(q, P, lam):
    n = q.size
    S = np.linalg.cholesky((n + lam) * (P + 1e-15 * np.eye(n)))   # tiny jitter keeps it PD
    return np.vstack([q, q + S.T, q - S.T])


class UKF:
    def __init__(self, mean0, cov0_diag, q1_si, beta_noise=config.BETA_NOISE, q2=config.Q2_DIAG):
        self.q = np.asarray(mean0, float).copy()
        self.P = np.diag(cov0_diag).astype(float)
        self.q1_si, self.beta_noise = q1_si, beta_noise
        self.lam, self.wm, self.wc = _weights(models.NX)
        self.r_inv = 1.0 / models.r_diag(q2)

    def step(self, m, x):
        # predict
        Q = np.diag(models.process_noise_std(self.q, self.q1_si, self.beta_noise) ** 2)
        X = models.g1(_sigma_points(self.q, self.P, self.lam))
        q_pred = self.wm @ X
        dX = X - q_pred
        P_pred = (dX.T * self.wc) @ dX + Q
        # update: redraw sigma points around the prediction, push through g2
        X = _sigma_points(q_pred, P_pred, self.lam)
        Z = np.array([models.g2(s, x) for s in X])
        z_pred = self.wm @ Z
        dX, dZ = X - q_pred, Z - z_pred
        Pxz = (dX.T * self.wc) @ dZ                                  # (4, L)
        # S^-1 Pxz^T via Woodbury, S = R + dZ^T diag(wc) dZ
        RinvD = self.r_inv[:, None] * dZ.T                           # (L, 9)
        core = np.diag(1.0 / self.wc) + dZ @ RinvD                   # (9, 9)
        SinvPzxT = self.r_inv[:, None] * Pxz.T - RinvD @ np.linalg.solve(core, RinvD.T @ Pxz.T)
        K = SinvPzxT.T                                               # (4, L) = Pxz S^-1
        self.q = q_pred + K @ (m - z_pred)
        self.P = P_pred - K @ Pxz.T
        self.P = (self.P + self.P.T) / 2
        return self.q
