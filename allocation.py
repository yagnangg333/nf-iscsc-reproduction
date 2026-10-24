"""Computing, power and vehicle assignment (Stage 5).

Models from the paper:
  P_Comp = -F sum_k xi ln(rho)                       Eq. (20)
  P_C&S  = Tr(W W^H + R R^H)                         Eq. (21)  (in beamforming.py)
  T      = (C D + L) / f                             Eqs. (22)-(23)
  L      = nu1 f1(RCRB(theta)) + nu2 f2(RCRB(d)) + nu3   Eq. (24), f1, f2 linear (Sec. VI)
  P_DT   = kappa f^3 C                               Eq. (25), read per config.DT_POWER
  Assignment: greedy (Alg. 2, lines 3-5), greedy with random flip, and the
  hybrid heuristic HH = greedy start + simulated annealing (Alg. 2).

Readings (docs/06-assumptions.md, docs/stage5-power-check.md):
  - F in mW and Eq. (25) as energy per slot (owner decision, 24 Oct 2026).
  - nu1 goes with RCRB(theta) in degrees and nu2 with RCRB(d) in metres, as the
    Table III units say (gap 24). |nu3| with nu3 ~ CN(0, 1000) (gap 25).
  - An RSU builds the DT only for the vehicles it serves (xi = 1). Alg. 2 says
    (34g)/(34h) can make an assignment infeasible, which needs this.
"""
import numpy as np

import config


# ------------------------------------------------------------------ power models
def f_coef_w(unit=config.F_UNIT):
    """F of Eq. (20) in watts."""
    return config.F_COMP * {"W": 1.0, "mW": 1e-3}[unit]


def comp_power(rho, unit=config.F_UNIT):
    """Eq. (20) for the served vehicles of one RSU: -F sum ln(rho). rho: array of served rhos."""
    return -f_coef_w(unit) * float(np.sum(np.log(np.asarray(rho, dtype=float))))


def dt_power(f, c, d, reading=config.DT_POWER):
    """DT computing power of one task (W) for CPU frequency f (Hz), C cycles/bit, D bits."""
    f, c, d = np.asarray(f, float), np.asarray(c, float), np.asarray(d, float)
    if reading == "as_written":
        return config.KAPPA * f**3 * c                       # Eq. (25)
    if reading == "energy_per_slot":
        return config.KAPPA * f**2 * c * d / config.DELTA_T  # energy kappa f^2 C D per slot
    if reading == "kappa_f3":
        return config.KAPPA * f**3
    raise ValueError(reading)


# --------------------------------------------------------------- latency models
def workload(rcrb_theta_deg, rcrb_d_m, nu3):
    """Extra cycles L of Eq. (24), linear f1 and f2 (Sec. VI)."""
    return config.NU1 * np.asarray(rcrb_theta_deg) + config.NU2 * np.asarray(rcrb_d_m) + np.abs(nu3)


def latency(c, d, l_extra, f):
    """Eqs. (22)-(23): T = C D / f + L / f (s)."""
    return (np.asarray(c) * np.asarray(d) + np.asarray(l_extra)) / np.asarray(f)


def min_frequency(c, d, l_extra, t_max=config.T_MAX):
    """Smallest f that meets (34g) for one task: f = (C D + L) / T^max.

    P_DT grows with f under every reading, and f appears nowhere else, so this is
    also the optimal f of problem (41) (it does not need the convex solver).
    """
    return (np.asarray(c) * np.asarray(d) + np.asarray(l_extra)) / t_max


def computing_plan(served, c, d, l_extra, reading=config.DT_POWER):
    """CPU frequencies and DT power of each RSU for one assignment.

    served: (M, K) 0/1 matrix xi. c, d, l_extra: (K,) per vehicle.
    Returns f (M, K), p_dt (M,), feasible (M,) for (34h) sum_k f <= F^max.
    """
    f_need = min_frequency(c, d, l_extra)
    f = served * f_need[None, :]
    p_dt = np.array([dt_power(f[m][served[m] > 0], c[served[m] > 0], d[served[m] > 0], reading).sum()
                     for m in range(served.shape[0])])
    feasible = f.sum(axis=1) <= config.F_MAX
    return f, p_dt, feasible


def draw_tasks(k, rng):
    """C (cycles/bit) and D (bits) per vehicle, uniform on the Table III ranges (gap 27)."""
    c = rng.uniform(*config.C_CYCLES_RANGE, size=k)
    d = rng.uniform(*config.D_BITS_RANGE, size=k)
    return c, d


def draw_nu3(k, rng):
    """|nu3| with nu3 ~ CN(0, 1000) (gap 25)."""
    z = (rng.standard_normal(k) + 1j * rng.standard_normal(k)) * np.sqrt(config.NU3_VAR / 2)
    return np.abs(z)


# ------------------------------------------------------------------ assignment
def to_matrix(assign, m=config.M_RSU):
    """Vector of RSU indices (K,) -> 0/1 matrix xi (M, K). Each column sums to 1 (34f)."""
    assign = np.asarray(assign)
    xi = np.zeros((m, assign.size), dtype=int)
    xi[assign, np.arange(assign.size)] = 1
    return xi


def greedy(dist):
    """Alg. 2, lines 3-5: each vehicle goes to its nearest RSU. dist: (M, K)."""
    return np.argmin(dist, axis=0)


def random_flip(assign, rng, p=config.FLIP_PROB, m=config.M_RSU):
    """Greedy with random flip: each vehicle moves to another RSU with probability p (gap 30)."""
    out = np.array(assign).copy()
    for k in np.flatnonzero(rng.random(out.size) < p):
        out[k] = rng.choice([j for j in range(m) if j != out[k]])
    return out


def neighbour(assign, rng, m=config.M_RSU):
    """SA neighbour: move one random vehicle to another RSU (gap 28)."""
    return random_flip_one(assign, rng.integers(len(assign)), rng, m)


def random_flip_one(assign, k, rng, m=config.M_RSU):
    out = np.array(assign).copy()
    out[k] = rng.choice([j for j in range(m) if j != out[k]])
    return out


def hybrid_heuristic(dist, solve, rng, t0=config.SA_T0, alpha=config.SA_ALPHA,
                     t_min=config.SA_TMIN, n_max=config.SA_NMAX):
    """Alg. 2: greedy start, then simulated annealing over assignments.

    solve(assign) -> score B (larger is better; B = -(41a), gap 29) or None if (41) is
    infeasible. Results are cached, so a revisited assignment costs nothing.
    Returns (best assignment, best score, history list of (n, B_n, B_best)).
    """
    cache = {}
    tabu = set()

    def score(a):
        key = tuple(int(v) for v in a)
        if key not in cache:
            cache[key] = solve(np.array(key))
        return cache[key]

    # Lines 1-10: greedy start; if it is infeasible (tabu), regenerate.
    current = greedy(dist)
    b_n = score(current)
    tries = 0
    while b_n is None and tries < n_max:
        tabu.add(tuple(current))
        current = rng.integers(0, dist.shape[0], size=dist.shape[1])
        if tuple(current) in tabu:
            continue
        b_n = score(current)
        tries += 1
    if b_n is None:
        return None, None, []
    # Line 11. B_best starts at -inf rather than 0, so a negative score can still win (gap 29).
    best, b_best = current.copy(), b_n
    t, n, history = t0, 0, []
    while n < n_max and t > t_min:                       # line 26
        new = neighbour(current, rng)                    # line 17
        if tuple(new) in tabu:
            t, n = alpha * t, n + 1
            continue
        b_new = score(new)                               # line 18
        if b_new is None:                                # lines 14-15
            tabu.add(tuple(new))
        elif b_new >= b_n or rng.random() <= np.exp((b_new - b_n) / t):   # line 19
            current, b_n = new, b_new                    # line 20
            if b_n >= b_best:                            # lines 21-22
                best, b_best = current.copy(), b_n
        t, n = alpha * t, n + 1                          # line 25
        history.append((n, b_n, b_best))
    return best, b_best, history
