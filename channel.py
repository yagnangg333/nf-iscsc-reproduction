"""Near-field and far-field array models.

Paper equations implemented here (Yang et al., IEEE TVT 2026, Sec. II-B):

  Eq. (5)  a^T(theta, d) = exp(-j 2pi/lambda * (-n_t d_t cos(theta) + n_t^2 d_t^2 sin^2(theta) / (2 d)))
           a^R(theta, d) = same form with n_r, d_r
           H(theta, d)   = exp(-j 2pi/(lambda d) * n_t d_t n_r d_r sin^2(theta))
           A^H(theta, d) = a^R (a^T)^H  (element-wise *)  H        -> size Nr x Nt
  Eq. (6)  Gamma[n_r, n_t] = 1 / sqrt(4 pi phi),
           phi = d^2 + (n_t d_t + n_r d_r)^2 - 2 d (n_t d_t + n_r d_r) cos(theta)
  Eq. (4)  y = Gamma A^H w x + interference + noise.
           ASSUMED (gap 9): "Gamma A^H" is element-wise, since both are Nr x Nt.

Not in the paper (ours, for the Stage 1 comparison only):
  exact_steering  - phase from the exact element distance (no Fresnel approximation)
  far_steering    - plane-wave model (the d terms dropped)

theta is measured from the array axis (ASSUMED, gap 2), so theta = pi/2 is broadside.
"""
import numpy as np

import config


def _offsets(n, spacing):
    """Element positions along the array axis, centred on the array (m)."""
    return config.centred_indices(n) * spacing


def nf_steering(theta, d, n=config.NT, spacing=config.D_T, lam=config.WAVELENGTH):
    """Near-field steering vector from Eq. (5) (Fresnel form). Returns shape (n,)."""
    delta = _offsets(n, spacing)
    path = -delta * np.cos(theta) + delta**2 * np.sin(theta) ** 2 / (2 * d)
    return np.exp(-1j * 2 * np.pi / lam * path)


def exact_steering(theta, d, n=config.NT, spacing=config.D_T, lam=config.WAVELENGTH):
    """Steering vector from the exact element distance r_n (ours, not in the paper).

    r_n = sqrt(d^2 + delta^2 - 2 d delta cos(theta)) by the law of cosines.
    Its second-order Taylor expansion in delta/d is exactly the phase of Eq. (5).
    """
    delta = _offsets(n, spacing)
    r = np.sqrt(d**2 + delta**2 - 2 * d * delta * np.cos(theta))
    return np.exp(-1j * 2 * np.pi / lam * (r - d))


def far_steering(theta, n=config.NT, spacing=config.D_T, lam=config.WAVELENGTH):
    """Plane-wave (far-field) steering vector: Eq. (5) with d -> infinity."""
    delta = _offsets(n, spacing)
    return np.exp(-1j * 2 * np.pi / lam * (-delta * np.cos(theta)))


def cross_term(theta, d, nt=config.NT, nr=config.NR, dt=config.D_T, dr=config.D_R,
               lam=config.WAVELENGTH):
    """Coupling term H(theta, d) of Eq. (5). Returns shape (nr, nt)."""
    pt = _offsets(nt, dt)
    pr = _offsets(nr, dr)
    return np.exp(-1j * 2 * np.pi / (lam * d) * np.outer(pr, pt) * np.sin(theta) ** 2)


def steering_matrix_H(theta, d, nt=config.NT, nr=config.NR, dt=config.D_T, dr=config.D_R,
                      lam=config.WAVELENGTH):
    """A^H(theta, d) of Eq. (5): a^R (a^T)^H, element-wise times H. Shape (nr, nt)."""
    a_t = nf_steering(theta, d, nt, dt, lam)
    a_r = nf_steering(theta, d, nr, dr, lam)
    return np.outer(a_r, a_t.conj()) * cross_term(theta, d, nt, nr, dt, dr, lam)


def path_loss_matrix(theta, d, nt=config.NT, nr=config.NR, dt=config.D_T, dr=config.D_R):
    """Gamma of Eq. (6), non-uniform spherical wave model. Shape (nr, nt).

    Note: phi is the squared distance between the two positions, so Gamma
    equals 1 / (sqrt(4 pi) * distance). The paper has no wavelength term here.
    """
    s = np.add.outer(_offsets(nr, dr), _offsets(nt, dt))     # n_t d_t + n_r d_r
    phi = d**2 + s**2 - 2 * d * s * np.cos(theta)
    return 1.0 / np.sqrt(4 * np.pi * phi)


def channel_matrix(theta, d, nt=config.NT, nr=config.NR, dt=config.D_T, dr=config.D_R,
                   lam=config.WAVELENGTH):
    """Downlink channel G = Gamma (element-wise) A^H, used in Eq. (4). Shape (nr, nt).

    The paper's H_{m,k,i} in Eq. (13) is G^H (shape nt x nr).
    """
    return path_loss_matrix(theta, d, nt, nr, dt, dr) * steering_matrix_H(theta, d, nt, nr, dt, dr, lam)


def received_signal(G, w_own, x_own, interf_beams, interf_symbols, sens_beams, sens_symbols,
                    sigma2_c, rng):
    """One draw of Eq. (4): desired + comms interference + sensing interference + noise.

    G: (nr, nt) channel of this vehicle; w_own: (nt,) its beam; x_own: its symbol.
    interf_beams: (nt, L) other comms beams (all RSUs); interf_symbols: (L,)
    sens_beams: (nt, S) sensing beams (all RSUs); sens_symbols: (S,)
    """
    nr = G.shape[0]
    noise = np.sqrt(sigma2_c / 2) * (rng.standard_normal(nr) + 1j * rng.standard_normal(nr))
    return (G @ (w_own * x_own)
            + G @ (interf_beams @ interf_symbols)
            + G @ (sens_beams @ sens_symbols)
            + noise)


def rayleigh_distance(n=config.NT, lam=config.WAVELENGTH):
    """Rayleigh distance for a half-wavelength ULA, using the brief's form N^2 lambda / 2."""
    return n**2 * lam / 2


def correlation(a, b):
    """Normalised correlation |a^H b| / N between two unit-modulus vectors."""
    return np.abs(np.vdot(a, b)) / a.size
