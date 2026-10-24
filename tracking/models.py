"""State and measurement models of the paper (Sec. II-C and II-D).

State model, Eq. (10) (taken from Liu et al. [31]):
  theta_i = theta_{i-1} + v_{i-1} dT sin(theta_{i-1}) / d_{i-1} + u_theta
  d_i     = d_{i-1} - v_{i-1} dT cos(theta_{i-1})              + u_d
  v_i     = v_{i-1}                                             + u_v
  beta_i  = beta_{i-1} (1 + v_{i-1} dT cos(theta_{i-1}) / d_{i-1}) + u_beta
Eq. (11): q_i = g1(q_{i-1}) + u_i,  u ~ N(0, Q1),  Q1 = diag(...) from Eq. (12) and Table III.

Measurement model, Eq. (9) and the text of Sec. II-C:
  z_hat = beta A(theta, d) A^H(theta, d) x + z_theta,  z_theta ~ CN(0, sigma1^2 I)   (Nt complex values)
  d_hat = c tau_hat / 2 + z_tau = d + z_tau,          z_tau ~ N(0, sigma2^2)
  v_hat = lambda mu_hat / 2 + z_mu = v + z_mu,        z_mu ~ N(0, sigma3^2)
  Remark 1 lets us skip the matched filter of Eq. (8): tau and mu are taken as exact.
Eq. (11): m_i = g2(q_i) + z_i,  Q2 = diag(sigma1^2 I, sigma2^2, sigma3^2) from Eq. (12).

For the filters we stack the measurement as real numbers:
  [Re(z_hat) (Nt), Im(z_hat) (Nt), d_hat, v_hat]  -> length 2 Nt + 2
so CN(0, sigma1^2) noise is N(0, sigma1^2 / 2) on each real part.
"""
import numpy as np

import channel
import config

NX = 4  # state size: theta, d, v, beta


# ------------------------------------------------------------------ state model
def g1(q, dt=config.DELTA_T):
    """Noise-free state transition of Eq. (10). q has shape (..., 4): theta, d, v, beta."""
    theta, d, v, beta = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty_like(q)
    out[..., 0] = theta + v * dt * np.sin(theta) / d
    out[..., 1] = d - v * dt * np.cos(theta)
    out[..., 2] = v
    out[..., 3] = beta * (1 + v * dt * np.cos(theta) / d)
    return out


def g1_jacobian(q, dt=config.DELTA_T):
    """d g1 / d q for one state q (shape (4,)). Used by the EKF."""
    th, d, v, b = q
    s, c = np.sin(th), np.cos(th)
    return np.array([
        [1 + v * dt * c / d, -v * dt * s / d**2, dt * s / d, 0.0],
        [v * dt * s, 1.0, -dt * c, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-b * v * dt * s / d, -b * v * dt * c / d**2, b * dt * c / d, 1 + v * dt * c / d],
    ])


def process_noise_std(q_prev, q1_si, beta_noise=config.BETA_NOISE):
    """Standard deviation of u_i for each state entry (shape like q_prev).

    "absolute": u_beta ~ N(0, sigma_beta^2), as Eq. (10)-(12) write it.
    "relative": u_beta ~ N(0, sigma_beta^2 beta_{i-1}^2) (our variant, not in the paper).
    """
    std = np.broadcast_to(np.sqrt(q1_si), q_prev.shape).copy()
    if beta_noise == "relative":
        std[..., 3] = np.sqrt(q1_si[3]) * np.abs(q_prev[..., 3])
    elif beta_noise != "absolute":
        raise ValueError(beta_noise)
    return std


def step(q_prev, rng, q1_si, beta_noise=config.BETA_NOISE, dt=config.DELTA_T):
    """One noisy step of Eq. (11): q_i = g1(q_{i-1}) + u_i."""
    return g1(q_prev, dt) + rng.standard_normal(q_prev.shape) * process_noise_std(q_prev, q1_si, beta_noise)


def propagate(q0, n_steps, rng=None, q1_si=None, beta_noise=config.BETA_NOISE, dt=config.DELTA_T):
    """Run Eq. (11) for n_steps. With rng=None (or q1_si=None) there is no noise.

    Returns shape (n_steps + 1, ..., 4), including the start state.
    """
    out = [np.asarray(q0, dtype=float)]
    for _ in range(n_steps):
        if rng is None or q1_si is None:
            out.append(g1(out[-1], dt))
        else:
            out.append(step(out[-1], rng, q1_si, beta_noise, dt))
    return np.stack(out)


# ------------------------------------------------------------ measurement model
def probe_beam(theta, d, power):
    """Transmit vector x for one vehicle: a near-field beam at (theta, d) with the given power.

    ASSUMED (gap 16): the RSU points x at the predicted vehicle state and gives each
    vehicle an equal share of Pt.
    """
    return np.sqrt(power / config.NT) * channel.nf_steering(theta, d)


def _phase_parts(theta, d, nr=config.NR):
    """Pieces of A^H = exp(-j k psi) (Eq. 5) for arrays of theta, d of shape (P,).

    Returns psi (P, Nr, Nt) and its derivatives w.r.t. theta and d.
    psi[r, t] = p_r - p_t + dt_t dr_r sin^2(theta) / d, with
    p(delta) = -delta cos(theta) + delta^2 sin^2(theta) / (2 d).
    """
    th = np.asarray(theta, float)[:, None, None]
    dd = np.asarray(d, float)[:, None, None]
    pt = (config.centred_indices(config.NT) * config.D_T)[None, None, :]
    pr = (config.centred_indices(nr) * config.D_R)[None, :, None]
    s, c = np.sin(th), np.cos(th)
    p_t = -pt * c + pt**2 * s**2 / (2 * dd)
    p_r = -pr * c + pr**2 * s**2 / (2 * dd)
    cross = pt * pr * s**2 / dd
    psi = p_r - p_t + cross
    # derivatives
    dp_t_th = pt * s + pt**2 * s * c / dd
    dp_r_th = pr * s + pr**2 * s * c / dd
    dpsi_th = dp_r_th - dp_t_th + 2 * pt * pr * s * c / dd
    dpsi_d = -(pr**2 - pt**2) * s**2 / (2 * dd**2) - pt * pr * s**2 / dd**2
    return psi, dpsi_th, dpsi_d


def _cis(phase):
    """exp(j phase) for a real array (cos + j sin is about twice as fast as np.exp)."""
    out = np.empty(phase.shape, complex)
    out.real, out.imag = np.cos(phase), np.sin(phase)
    return out


def _ah_parts(q):
    """Pieces of the rows of A^H (Eq. 5) for states q (P, 4).

    A^H[r, t] = a_R[r] conj(a_T[t]) H[r, t], with
      conj(a_T[t]) = exp(+jk p_t),  a_R[r] = exp(-jk p_r),
      H[r, t]      = exp(-jk delta_r delta_t sin^2(theta) / d),
      p(delta)     = -delta cos(theta) + delta^2 sin^2(theta) / (2 d).
    Returns (aT_conj (P, Nt), list of a_R[r] (P, 1), list of H rows (P, Nt) or None for delta_r = 0).
    The receive offsets come in +/- pairs, so the H row for -delta_r is the conjugate
    of the one for +delta_r; we reuse it.
    """
    k = 2 * np.pi / config.WAVELENGTH
    th, d = q[:, 0:1], q[:, 1:2]
    s2, c = np.sin(th) ** 2, np.cos(th)
    pt = config.centred_indices(config.NT) * config.D_T
    pr = config.centred_indices(config.NR) * config.D_R
    aT_conj = _cis(k * (-pt * c + pt**2 * s2 / (2 * d)))
    a_r, h_rows, cache = [], [], {}
    for r in pr:
        a_r.append(_cis(-k * (-r * c + r**2 * s2 / (2 * d))))
        if r == 0:
            h_rows.append(None)
            continue
        if abs(r) not in cache:
            cache[abs(r)] = _cis(-k * abs(r) * pt * s2 / d)
        h_rows.append(cache[abs(r)] if r > 0 else cache[abs(r)].conj())
    return aT_conj, a_r, h_rows


def _ah_rows(q):
    """Rows of A^H for states q (P, 4): list of Nr arrays, each (P, Nt)."""
    aT_conj, a_r, h_rows = _ah_parts(q)
    return [a * (aT_conj if h is None else aT_conj * h) for a, h in zip(a_r, h_rows)]


def _gram(q):
    """A^H A for states q (P, 4), shape (P, Nr, Nr), in closed form.

    (A^H A)[r, r'] = sum_t A^H[r, t] conj(A^H[r', t]); the conj(a_T) parts cancel and the
    sum over t of exp(-j u (t - (Nt-1)/2)) is the Dirichlet kernel sin(Nt u / 2) / sin(u / 2),
    with u = k (delta_r - delta_r') d_t sin^2(theta) / d.
    """
    k = 2 * np.pi / config.WAVELENGTH
    th, d = q[:, 0], q[:, 1]
    s2, c = np.sin(th) ** 2, np.cos(th)
    pr = config.centred_indices(config.NR) * config.D_R
    a = np.stack([_cis(-k * (-r * c + r**2 * s2 / (2 * d))) for r in pr], axis=1)   # (P, Nr)
    u = k * (pr[:, None] - pr[None, :])[None] * config.D_T * (s2 / d)[:, None, None]
    half = u / 2
    small = np.abs(np.sin(half)) < 1e-12
    dirich = np.where(small, config.NT, np.sin(config.NT * half) / np.where(small, 1.0, np.sin(half)))
    return a[:, :, None] * a.conj()[:, None, :] * dirich


def echo_mean(q, x):
    """Noise-free echo beta A A^H x of Eq. (9) for states q (P, 4) and beam x (Nt,). Shape (P, Nt)."""
    mu = np.zeros((q.shape[0], config.NT), complex)
    for row in _ah_rows(q):
        mu += row.conj() * (row @ x)[:, None]                         # A[:, r] (A^H x)[r]
    return q[:, 3:4] * mu


def g2(q, x):
    """Stacked real measurement of Eq. (11) for one state q (4,). Length 2 Nt + 2."""
    mu = echo_mean(q[None, :], x)[0]
    return np.concatenate([mu.real, mu.imag, [q[1], q[2]]])


def g2_jacobian(q, x):
    """Analytic d g2 / d q for one state (shape (2 Nt + 2, 4)). Used by the EKF."""
    k = 2 * np.pi / config.WAVELENGTH
    psi, dth, dd = _phase_parts(q[None, 0], q[None, 1])
    AH = np.exp(-1j * k * psi)[0]                    # (Nr, Nt)
    dAH_th = -1j * k * dth[0] * AH
    dAH_d = -1j * k * dd[0] * AH
    y = AH @ x
    beta = q[3]

    def d_mu(dAH):  # derivative of A (A^H x) for one parameter
        return dAH.conj().T @ y + AH.conj().T @ (dAH @ x)

    J = np.zeros((2 * config.NT + 2, NX))
    for col, dAH in ((0, dAH_th), (1, dAH_d)):
        dm = beta * d_mu(dAH)
        J[:config.NT, col], J[config.NT:2 * config.NT, col] = dm.real, dm.imag
    m0 = AH.conj().T @ y                             # d mu / d beta
    J[:config.NT, 3], J[config.NT:2 * config.NT, 3] = m0.real, m0.imag
    J[2 * config.NT, 1] = 1.0                        # d_hat depends on d
    J[2 * config.NT + 1, 2] = 1.0                    # v_hat depends on v
    return J


def r_diag(q2=config.Q2_DIAG):
    """Diagonal of the stacked measurement covariance (real parts get sigma1^2 / 2)."""
    s1, s2, s3 = q2
    return np.concatenate([np.full(2 * config.NT, s1 / 2), [s2, s3]])


def measure(q, x, rng, q2=config.Q2_DIAG):
    """One noisy stacked measurement m = g2(q) + z for state q (4,)."""
    return g2(q, x) + rng.standard_normal(2 * config.NT + 2) * np.sqrt(r_diag(q2))


def log_likelihood(m, qs, x, q2=config.Q2_DIAG):
    """log p(m | q) (up to a constant) for many states qs (P, 4). Used by the PF.

    |z - mu|^2 is expanded as |z|^2 - 2 Re(z^H mu) + |mu|^2 with mu = beta A y,
    y = A^H x, so we never build mu itself:
      z^H mu  = beta sum_r y_r conj(row_r . z)
      |mu|^2  = beta^2 y^H (A^H A) y
    (row_r is row r of A^H). Same value as the direct formula; tested.
    """
    s1, s2, s3 = q2
    z = m[:config.NT] + 1j * m[config.NT:2 * config.NT]
    aT_conj, a_r, h_rows = _ah_parts(qs)
    y, rz = [], []
    for a, h in zip(a_r, h_rows):
        row = aT_conj if h is None else aT_conj * h                   # row_r / a_R[r]
        y.append(a[:, 0] * (row @ x))
        rz.append(a[:, 0] * (row @ z))
    y, rz = np.stack(y, axis=1), np.stack(rz, axis=1)                 # (P, Nr)
    beta = qs[:, 3]
    cross = beta * np.sum(y * rz.conj(), axis=1)
    mu2 = beta**2 * np.real(np.einsum("pa,pab,pb->p", y.conj(), _gram(qs), y))
    ll = -(np.vdot(z, z).real - 2 * cross.real + mu2) / s1
    ll -= (m[2 * config.NT] - qs[:, 1]) ** 2 / (2 * s2)
    ll -= (m[2 * config.NT + 1] - qs[:, 2]) ** 2 / (2 * s3)
    return ll
