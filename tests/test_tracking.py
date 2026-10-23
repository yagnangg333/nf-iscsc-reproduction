"""Checks for the state model, measurement model and the three filters (Stage 3)."""
import numpy as np
import pytest

import channel
import config
from tracking import models
from tracking.ekf import EKF
from tracking.pf import ParticleFilter
from tracking.ukf import UKF

Q_TEST = np.array([0.9, 30.0, 20.0, config.BETA_SCALE / 30.0])
X_BEAM = models.probe_beam(0.9, 30.0, config.PT_W / 5)


# ---------------------------------------------------------------- noise units
def test_q1_unit_conversion():
    np.testing.assert_allclose(config.q1_si("rad2"), config.Q1_DIAG)
    q = config.q1_si("deg2")
    assert q[0] == pytest.approx(0.02 * (np.pi / 180) ** 2)
    np.testing.assert_allclose(q[1:], config.Q1_DIAG[1:])
    with pytest.raises(ValueError):
        config.q1_si("grad")


# ---------------------------------------------------------------- state model
def test_zero_noise_step_is_g1():
    rng = np.random.default_rng(0)
    q = models.step(Q_TEST, rng, np.zeros(4))
    np.testing.assert_allclose(q, models.g1(Q_TEST))


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_process_noise_statistics(mode):
    # u = q_i - g1(q_{i-1}) must have zero mean and the variances of Eq. (12).
    rng = np.random.default_rng(1)
    q1 = config.q1_si("deg2")
    qs = np.tile(Q_TEST, (200_000, 1))
    u = models.step(qs, rng, q1, mode) - models.g1(qs)
    target = q1.copy()
    if mode == "relative":
        target[3] = q1[3] * Q_TEST[3] ** 2
    assert np.all(np.abs(u.mean(axis=0)) < 4 * np.sqrt(target / 200_000))   # within 4 standard errors
    np.testing.assert_allclose(u.var(axis=0), target, rtol=0.02)


def test_truth_is_reproducible_with_seed():
    q1 = config.q1_si()
    a = models.propagate(Q_TEST, 30, np.random.default_rng(5), q1)
    b = models.propagate(Q_TEST, 30, np.random.default_rng(5), q1)
    c = models.propagate(Q_TEST, 30, np.random.default_rng(6), q1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_g1_jacobian_matches_finite_differences():
    eps = 1e-6 * np.maximum(np.abs(Q_TEST), 1e-3)
    num = np.column_stack([(models.g1(Q_TEST + e) - models.g1(Q_TEST - e)) / (2 * e[i])
                           for i, e in enumerate(np.diag(eps))])
    np.testing.assert_allclose(models.g1_jacobian(Q_TEST), num, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------- measurement model
def test_echo_matches_channel_module():
    # beta A A^H x built from channel.steering_matrix_H (Eq. 5) directly.
    AH = channel.steering_matrix_H(Q_TEST[0], Q_TEST[1])
    direct = Q_TEST[3] * AH.conj().T @ (AH @ X_BEAM)
    np.testing.assert_allclose(models.echo_mean(Q_TEST[None], X_BEAM)[0], direct, atol=1e-15)


def test_gram_closed_form():
    qs = np.array([[0.4, 12.0, 20, 1.0], [1.6, 55.0, 20, 1.0], [2.9, 80.0, 20, 1.0]])
    for q, G in zip(qs, models._gram(qs)):
        AH = channel.steering_matrix_H(q[0], q[1])
        np.testing.assert_allclose(G, AH @ AH.conj().T, atol=1e-9)


def test_log_likelihood_matches_direct_formula():
    rng = np.random.default_rng(2)
    m = models.measure(Q_TEST, X_BEAM, rng)
    qs = Q_TEST + rng.standard_normal((20, 4)) * [1e-3, 0.1, 0.5, 1e-4]
    mu = models.echo_mean(qs, X_BEAM)
    z = m[:config.NT] + 1j * m[config.NT:2 * config.NT]
    s1, s2, s3 = config.Q2_DIAG
    direct = (-np.sum(np.abs(z - mu) ** 2, axis=1) / s1
              - (m[-2] - qs[:, 1]) ** 2 / (2 * s2) - (m[-1] - qs[:, 2]) ** 2 / (2 * s3))
    np.testing.assert_allclose(models.log_likelihood(m, qs, X_BEAM), direct, rtol=1e-9)


def test_measurement_noise_statistics():
    # zeta ~ CN(0, sigma1^2) -> each real part N(0, sigma1^2/2); d, v noise N(0, sigma2^2), N(0, sigma3^2)
    rng = np.random.default_rng(3)
    clean = models.g2(Q_TEST, X_BEAM)
    noise = np.array([models.measure(Q_TEST, X_BEAM, rng) - clean for _ in range(4000)])
    np.testing.assert_allclose(noise.var(axis=0).reshape(-1)[[0, 400, 620, 621]],
                               models.r_diag()[[0, 400, 620, 621]], rtol=0.1)
    assert noise[:, :2 * config.NT].var() == pytest.approx(config.Q2_DIAG[0] / 2, rel=0.01)


def test_g2_jacobian_matches_finite_differences():
    J = models.g2_jacobian(Q_TEST, X_BEAM)
    eps = np.array([1e-7, 1e-5, 1e-5, 1e-7])
    num = np.column_stack([(models.g2(Q_TEST + e, X_BEAM) - models.g2(Q_TEST - e, X_BEAM)) / (2 * eps[i])
                           for i, e in enumerate(np.diag(eps))])
    np.testing.assert_allclose(J, num, rtol=1e-4, atol=1e-6 * np.abs(num).max())


# ------------------------------------------------------------------ filters
def _run(filter_name, n_steps=40, seed=4):
    """Track one car with (almost) no noise from a slightly wrong start."""
    tiny_q1 = np.array([1e-12, 1e-10, 1e-10, 1e-16])
    tiny_q2 = (1e-6, 1e-6, 1e-6)
    rng = np.random.default_rng(seed)
    truth = models.propagate(Q_TEST, n_steps)
    start = Q_TEST + np.array([np.deg2rad(0.05), 0.05, 0.1, 0.0])
    cov0 = np.array([np.deg2rad(0.1) ** 2, 0.1**2, 0.2**2, 1e-14])
    if filter_name == "pf":
        f = ParticleFilter(start, cov0, 2000, np.random.default_rng(seed), tiny_q1, "absolute", tiny_q2)
    else:
        f = (EKF if filter_name == "ekf" else UKF)(start, cov0, tiny_q1, "absolute", tiny_q2)
    for i in range(1, n_steps + 1):
        x = models.probe_beam(*models.g1(truth[i - 1])[:2], config.PT_W / 5)
        est = f.step(models.measure(truth[i], x, rng, tiny_q2), x)
    return est - truth[-1]


@pytest.mark.parametrize("name", ["ekf", "ukf"])
def test_kalman_filters_converge_without_noise(name):
    err = _run(name)
    assert abs(err[0]) < np.deg2rad(1e-3) and abs(err[1]) < 1e-3 and abs(err[2]) < 1e-2


def test_particle_filter_converges_without_noise():
    # With no process noise the PF can only pick among its starting particles,
    # so it ends near (not exactly at) the truth.
    err = _run("pf")
    assert abs(err[0]) < np.deg2rad(0.05) and abs(err[1]) < 0.05 and abs(err[2]) < 0.1


def test_pf_weights_and_resampling():
    rng = np.random.default_rng(7)
    f = ParticleFilter(Q_TEST, config.q1_si(), 500, rng, config.q1_si(), "relative")
    x = X_BEAM
    f.step(models.measure(models.g1(Q_TEST), x, rng), x)
    assert f.particles.shape == (500, 4)
    np.testing.assert_allclose(f.weights.sum(), 1.0)
    assert 1.0 <= f.ess <= 500
