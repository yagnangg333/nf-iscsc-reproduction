"""Particle filter, Algorithm 1 of the paper (Sec. IV-B).

Eq. (29): posterior ~ sum_n w_n delta(q - q_n)
Eq. (31): q_n,i ~ p(q_i | q_n,i-1)              (draw from the state model, Eq. 11)
Eq. (32): w_n,i = w_n,i-1 p(m_i | q_n,i)          (likelihood from the measurement model)
Eq. (33): normalise the weights
Alg. 1 line 9: multinomial resampling every slot, so w_n,i-1 = 1/Ns after each resample.
Eq. (30): estimate = sum_n w_n q_n. We take it with the normalised weights before
          resampling; after resampling it would be the plain particle mean, which is
          the same in expectation but noisier.
"""
import numpy as np

import config
from tracking import models


class ParticleFilter:
    def __init__(self, mean0, cov0_diag, n_particles, rng, q1_si, beta_noise=config.BETA_NOISE,
                 q2=config.Q2_DIAG):
        self.rng = rng
        self.q1_si, self.beta_noise, self.q2 = q1_si, beta_noise, q2
        self.n = n_particles
        # Alg. 1 lines 1-2: draw particles from the prior, equal weights.
        self.particles = mean0 + rng.standard_normal((n_particles, models.NX)) * np.sqrt(cov0_diag)
        self.weights = np.full(n_particles, 1.0 / n_particles)
        self.estimate = np.asarray(mean0, float).copy()
        self.ess = float(n_particles)

    def step(self, m, x):
        """One slot: predict (31), weight (32), normalise (33), estimate (30), resample."""
        self.particles = models.step(self.particles, self.rng, self.q1_si, self.beta_noise)
        logw = np.log(self.weights) + models.log_likelihood(m, self.particles, x, self.q2)
        logw -= logw.max()                     # avoid underflow before exp
        w = np.exp(logw)
        w /= w.sum()
        self.estimate = w @ self.particles
        self.ess = 1.0 / np.sum(w**2)          # effective sample size, for diagnostics only
        idx = self.rng.choice(self.n, self.n, p=w)
        self.particles = self.particles[idx]
        self.weights = np.full(self.n, 1.0 / self.n)
        return self.estimate
