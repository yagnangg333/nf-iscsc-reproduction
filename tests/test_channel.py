"""Checks for channel.py against known values, limits and symmetries."""
import numpy as np
import pytest

import channel
import config

ANGLES = np.deg2rad([30.0, 60.0, 90.0, 120.0])


def test_wavelength_and_spacing():
    # 50 GHz -> lambda = c / f = 5.996 mm, spacing = lambda / 2
    assert config.WAVELENGTH == pytest.approx(5.9958e-3, rel=1e-4)
    assert config.D_T == pytest.approx(config.WAVELENGTH / 2)


def test_rayleigh_distance_matches_formula():
    # Brief: Nt^2 * lambda / 2 with Nt = 310 at 50 GHz -> 288.1 m
    assert channel.rayleigh_distance() == pytest.approx(310**2 * config.WAVELENGTH / 2)
    assert channel.rayleigh_distance() == pytest.approx(288.1, abs=0.1)


@pytest.mark.parametrize("theta", ANGLES)
def test_steering_vectors_unit_modulus(theta):
    for a in (channel.nf_steering(theta, 20.0), channel.exact_steering(theta, 20.0),
              channel.far_steering(theta)):
        assert a.shape == (config.NT,)
        np.testing.assert_allclose(np.abs(a), 1.0, atol=1e-12)


@pytest.mark.parametrize("theta", ANGLES)
def test_near_field_tends_to_far_field(theta):
    # As d -> infinity the quadratic (distance) term vanishes.
    far = channel.far_steering(theta)
    assert channel.correlation(channel.nf_steering(theta, 1e7), far) > 0.99999
    assert channel.correlation(channel.exact_steering(theta, 1e7), far) > 0.99999


@pytest.mark.parametrize("theta", ANGLES)
def test_fresnel_matches_exact_far_from_array(theta):
    # Eq. (5) is the 2nd-order expansion of the exact distance, so the two
    # agree when d >> aperture (0.93 m) and differ close to the array.
    corr_far = channel.correlation(channel.nf_steering(theta, 50.0), channel.exact_steering(theta, 50.0))
    assert corr_far > 0.999


def test_fresnel_breaks_down_very_close():
    theta = np.deg2rad(60.0)
    corr = channel.correlation(channel.nf_steering(theta, 1.0), channel.exact_steering(theta, 1.0))
    assert corr < 0.9


def test_endfire_has_no_near_field_term():
    # sin(0) = 0 removes the distance term in Eq. (5), at any distance.
    np.testing.assert_allclose(channel.nf_steering(0.0, 3.0), channel.far_steering(0.0), atol=1e-12)


def test_mirror_symmetry():
    # theta -> pi - theta mirrors the array: element n becomes element -n.
    theta, d = np.deg2rad(50.0), 15.0
    a = channel.nf_steering(theta, d)
    b = channel.nf_steering(np.pi - theta, d)
    np.testing.assert_allclose(b, a[::-1], atol=1e-10)


def test_near_field_differs_from_far_field_close_in():
    # At broadside and 10 m (well inside 288 m) the far-field beam should miss badly.
    theta = np.pi / 2
    assert channel.correlation(channel.exact_steering(theta, 10.0), channel.far_steering(theta)) < 0.5


def test_steering_matrix_shape_and_structure():
    theta, d = np.deg2rad(70.0), 25.0
    AH = channel.steering_matrix_H(theta, d)
    assert AH.shape == (config.NR, config.NT)
    np.testing.assert_allclose(np.abs(AH), 1.0, atol=1e-12)
    # With sin(theta) = 0 the cross term H is all ones, so A^H is a plain outer product.
    AH0 = channel.steering_matrix_H(0.0, d)
    outer = np.outer(channel.nf_steering(0.0, d, config.NR, config.D_R), channel.nf_steering(0.0, d).conj())
    np.testing.assert_allclose(AH0, outer, atol=1e-12)


def test_path_loss_is_inverse_geometric_distance():
    # phi in Eq. (6) is the squared distance (law of cosines), so
    # Gamma = 1 / (sqrt(4 pi) * distance). Check against plain geometry.
    theta, d = np.deg2rad(40.0), 12.0
    G = channel.path_loss_matrix(theta, d)
    s = np.add.outer(config.centred_indices(config.NR) * config.D_R,
                     config.centred_indices(config.NT) * config.D_T)
    target = d * np.array([np.cos(theta), np.sin(theta)])
    dist = np.hypot(target[0] - s, target[1])
    np.testing.assert_allclose(G, 1 / (np.sqrt(4 * np.pi) * dist), rtol=1e-12)


def test_received_signal_noiseless_single_user():
    # With no interference and no noise, Eq. (4) reduces to y = G w x.
    rng = np.random.default_rng(0)
    theta, d = np.deg2rad(80.0), 30.0
    G = channel.channel_matrix(theta, d)
    w = channel.nf_steering(theta, d) / np.sqrt(config.NT)
    empty = np.zeros((config.NT, 0))
    y = channel.received_signal(G, w, 1.0, empty, np.zeros(0), empty, np.zeros(0), 0.0, rng)
    np.testing.assert_allclose(y, G @ w, atol=1e-15)
