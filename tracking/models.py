"""Motion model of the paper (state transition g1).

Eq. (10), Yang et al. 2026 (taken from Liu et al. [31]):
  theta_i = theta_{i-1} + v_{i-1} dT sin(theta_{i-1}) / d_{i-1} + u_theta
  d_i     = d_{i-1} - v_{i-1} dT cos(theta_{i-1})              + u_d
  v_i     = v_{i-1}                                             + u_v
  beta_i  = beta_{i-1} (1 + v_{i-1} dT cos(theta_{i-1}) / d_{i-1}) + u_beta
Eq. (11): q_i = g1(q_{i-1}) + u_i, with u ~ N(0, Q1), Q1 from Eq. (12).

This is a first-order (one Euler step) model of a car moving towards -x past
an RSU whose array lies along +x. The measurement model g2 comes in Stage 3.
"""
import numpy as np

import config


def g1(q, dt=config.DELTA_T):
    """Noise-free state transition of Eq. (10). q has shape (..., 4): theta, d, v, beta."""
    theta, d, v, beta = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty_like(q)
    out[..., 0] = theta + v * dt * np.sin(theta) / d
    out[..., 1] = d - v * dt * np.cos(theta)
    out[..., 2] = v
    out[..., 3] = beta * (1 + v * dt * np.cos(theta) / d)
    return out


def propagate(q0, n_steps, dt=config.DELTA_T, q1_diag=None, rng=None):
    """Run Eq. (10) for n_steps. With q1_diag and rng, add process noise u ~ N(0, Q1).

    Returns shape (n_steps + 1, ..., 4), including the start state.
    """
    out = [np.asarray(q0, dtype=float)]
    for _ in range(n_steps):
        q = g1(out[-1], dt)
        if q1_diag is not None:
            q = q + rng.standard_normal(q.shape) * np.sqrt(np.asarray(q1_diag))
        out.append(q)
    return np.stack(out)
