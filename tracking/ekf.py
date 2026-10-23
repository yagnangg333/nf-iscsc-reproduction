"""Extended Kalman filter on the same state model (Eq. 10-12) and measurement model (Eq. 9).

The paper cites [2] for its EKF and does not write the steps out, so we use the
textbook EKF:
  predict:  q- = g1(q),            P- = F P F^T + Q,      F = d g1 / d q
  update:   H = d g2 / d q at q-,  P+ = (P-^-1 + H^T R^-1 H)^-1
            q+ = q- + P+ H^T R^-1 (m - g2(q-))
The update is written in information form. It is algebraically the same as the
usual gain form, but inverts a 4 x 4 matrix instead of a (2 Nt + 2)-square one,
because R is diagonal.
"""
import numpy as np

import config
from tracking import models


class EKF:
    def __init__(self, mean0, cov0_diag, q1_si, beta_noise=config.BETA_NOISE, q2=config.Q2_DIAG):
        self.q = np.asarray(mean0, float).copy()
        self.P = np.diag(cov0_diag).astype(float)
        self.q1_si, self.beta_noise = q1_si, beta_noise
        self.r_inv = 1.0 / models.r_diag(q2)

    def step(self, m, x):
        Q = np.diag(models.process_noise_std(self.q, self.q1_si, self.beta_noise) ** 2)
        F = models.g1_jacobian(self.q)
        q_pred = models.g1(self.q)
        P_pred = F @ self.P @ F.T + Q
        H = models.g2_jacobian(q_pred, x)
        info = np.linalg.inv(P_pred) + H.T @ (self.r_inv[:, None] * H)
        self.P = np.linalg.inv(info)
        self.P = (self.P + self.P.T) / 2
        self.q = q_pred + self.P @ (H.T @ (self.r_inv * (m - models.g2(q_pred, x))))
        return self.q
