"""Checks for crb.py: derivatives, FIM properties, scaling laws, and CRB vs Monte Carlo ML."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

import config
import crb
from tracking import models

TH, D = np.deg2rad(60.0), 30.0
X1 = models.probe_beam(TH, D, config.PT_W)[:, None]          # one beam, Rx = x x^H
BETA = config.BETA_SCALE / D * np.exp(0.7j)


def test_b_derivatives_match_finite_differences():
    _, dth, dd = crb.b_products(TH, D, X1)
    h_th, h_d = 1e-7, 1e-5
    num_th = (crb.b_products(TH + h_th, D, X1)[0] - crb.b_products(TH - h_th, D, X1)[0]) / (2 * h_th)
    num_d = (crb.b_products(TH, D + h_d, X1)[0] - crb.b_products(TH, D - h_d, X1)[0]) / (2 * h_d)
    np.testing.assert_allclose(dth, num_th, rtol=1e-5, atol=1e-6 * np.abs(num_th).max())
    np.testing.assert_allclose(dd, num_d, rtol=1e-5, atol=1e-6 * np.abs(num_d).max())


def test_b_matches_full_matrix():
    AH = crb._ah_with_derivs(TH, D)[0]
    np.testing.assert_allclose(crb.b_products(TH, D, X1)[0], AH.conj().T @ AH @ X1, atol=1e-12)


def test_fim_symmetric_positive_definite():
    J = crb.fim_paper(TH, D, BETA, X1)
    np.testing.assert_allclose(J, J.T)
    assert np.all(np.linalg.eigvalsh(J) > 0)


def test_crb_scaling_laws():
    cd, ct = crb.crb_paper(TH, D, BETA, X1)
    # T samples -> 1/T; twice the noise -> twice the CRB; twice the power -> half
    np.testing.assert_allclose(crb.crb_paper(TH, D, BETA, X1, T=10), (cd / 10, ct / 10), rtol=1e-9)
    np.testing.assert_allclose(crb.crb_paper(TH, D, BETA, X1, sigma2=2 * config.SIGMA2_R_W), (2 * cd, 2 * ct), rtol=1e-9)
    np.testing.assert_allclose(crb.crb_paper(TH, D, BETA, np.sqrt(2) * X1), (cd / 2, ct / 2), rtol=1e-9)


def test_crb_does_not_depend_on_beta_phase():
    a = crb.crb_paper(TH, D, abs(BETA), X1)
    b = crb.crb_paper(TH, D, abs(BETA) * np.exp(2.1j), X1)
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_distance_harder_far_away():
    # Near-field distance information fades with range (far field: none at all).
    near = crb.crb_paper(TH, 20.0, config.BETA_SCALE / 20.0, models.probe_beam(TH, 20.0, 1.0)[:, None])[0]
    far = crb.crb_paper(TH, 200.0, config.BETA_SCALE / 20.0, models.probe_beam(TH, 200.0, 1.0)[:, None])[0]
    assert far > 50 * near


def _echo(theta, d, x):
    return crb.b_products(theta, d, x[:, None])[0][:, 0]


def test_crb_matches_ml_for_angle_with_known_d_and_beta():
    # One unknown (theta). At high SNR the ML estimate is efficient: MSE ~ CRB = 1 / J_thth.
    rng = np.random.default_rng(0)
    x = X1[:, 0]
    sigma2 = 1e-3
    J = crb.fim_paper(TH, D, BETA, X1, sigma2)
    crb_theta = 1.0 / J[1, 1]
    clean = BETA * _echo(TH, D, x)
    err = []
    for _ in range(1500):
        z = clean + np.sqrt(sigma2 / 2) * (rng.standard_normal(config.NT) + 1j * rng.standard_normal(config.NT))
        w = 30 * np.sqrt(crb_theta)
        res = minimize_scalar(lambda t: np.sum(np.abs(z - BETA * _echo(t, D, x)) ** 2),
                              bounds=(TH - w, TH + w), method="bounded", options=dict(xatol=1e-12))
        err.append(res.x - TH)
    ratio = np.mean(np.square(err)) / crb_theta
    assert 0.85 < ratio < 1.15


def test_crb_matches_ml_jointly_with_unknown_beta():
    # Unknowns theta, d and complex beta (Eq. 19 with the Schur complement).
    # beta is solved in closed form; (theta, d) by Nelder-Mead from the truth.
    rng = np.random.default_rng(1)
    x = X1[:, 0]
    sigma2 = 1e-4
    cd, ct = crb.crb_paper(TH, D, BETA, X1, sigma2)
    clean = BETA * _echo(TH, D, x)
    scale = np.array([np.sqrt(ct), np.sqrt(cd)])

    def cost(p, z):
        th, dd = TH + p[0] * scale[0], D + p[1] * scale[1]
        e = _echo(th, dd, x)
        return -np.abs(np.vdot(e, z)) ** 2 / np.vdot(e, e).real

    err = []
    for _ in range(400):
        z = clean + np.sqrt(sigma2 / 2) * (rng.standard_normal(config.NT) + 1j * rng.standard_normal(config.NT))
        res = minimize(cost, [0.0, 0.0], args=(z,), method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-12, maxiter=400))
        err.append(res.x * scale)
    err = np.array(err)
    assert 0.75 < np.mean(err[:, 0] ** 2) / ct < 1.3
    assert 0.75 < np.mean(err[:, 1] ** 2) / cd < 1.3


def test_posterior_crb_bounds():
    # More information than one snapshot; equals the snapshot CRB when the motion model
    # carries no information (huge process noise).
    q0 = np.array([TH, D, 20.0, config.BETA_SCALE / D])
    track = models.propagate(q0, 5)[None]
    beams = np.array([[models.probe_beam(*models.g1(track[0, i - 1])[:2], config.PT_W / 5) for i in range(1, 6)]])
    q1 = config.q1_si()
    p = crb.posterior_crb(track, beams, q1, q1, beta_noise="absolute")
    snap = crb.snapshot_crb(track[0, -1], beams[0, -1])
    assert np.all(p[-1] <= snap * (1 + 1e-9))
    huge = crb.posterior_crb(track, beams, q1 * 1e12, q1 * 1e12, beta_noise="absolute")
    np.testing.assert_allclose(huge[-1], snap, rtol=1e-3)
