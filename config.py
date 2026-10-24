"""All simulation parameters in one place.

Source tags:
  T-III   = Table III of Yang et al., IEEE TVT 2026 (p. 21003)
  Eq. (n) = equation n of the paper
  Sec. X  = section X of the paper
  ASSUMED = not given in the paper; reason in the comment (see docs/06-assumptions.md)

SI units everywhere: metres, seconds, hertz, watts, radians.
"""
import numpy as np

# ---------------- physical constants ----------------
C_LIGHT = 299_792_458.0          # speed of light, m/s

def dbm_to_w(p_dbm):
    """Convert dBm to watts."""
    return 10 ** (p_dbm / 10) / 1000.0

# ---------------- radio and arrays ----------------
FC = 50e9                        # carrier frequency, Hz (T-III, "f = 50 GHz")
WAVELENGTH = C_LIGHT / FC        # lambda, m (about 5.996 mm)
NT = 310                         # RSU transmit antennas (T-III)
NR = 3                           # vehicle receive antennas (T-III)
D_T = WAVELENGTH / 2             # RSU antenna spacing, m (Sec. VI: half-wavelength)
D_R = WAVELENGTH / 2             # vehicle antenna spacing, m (Sec. VI: half-wavelength)
M_RSU = 2                        # number of RSUs (T-III)
K_DEFAULT = 5                    # default number of vehicles (T-III)
K_SWEEP = (5, 10, 15, 20)        # vehicle counts in Fig. 3 (Figs. 5-7 use 5..20)

# ASSUMED (gap 3): antenna indices are centred on the array, because the paper
# measures d and theta from the array centre (Sec. II-B). With NT even the
# indices are half-integers: -154.5, ..., +154.5.
def centred_indices(n):
    """Element indices n - (N-1)/2 for an N-element ULA."""
    return np.arange(n) - (n - 1) / 2

# ASSUMED (gap 2): theta is measured from the array axis (= road direction),
# so theta = 90 deg is broadside. Eq. (5) and Eq. (10) only fit this convention.

# ---------------- powers and noise ----------------
PT_DBM = 25.0                    # RSU power budget, dBm (T-III)
PT_W = dbm_to_w(PT_DBM)          # about 0.316 W
SIGMA2_C_W = dbm_to_w(-30.0)     # comms noise power sigma_c^2, W (T-III)
SIGMA2_R_W = dbm_to_w(-30.0)     # echo noise power sigma_r^2, W (T-III)

# ---------------- timing ----------------
DELTA_T = 0.02                   # time slot length, s (T-III)
T_MAX = 0.015                    # max processing latency, s (T-III)
N_SLOTS = 100                    # ASSUMED (gap 8): 2 s horizon, as in Fig. 4

# ---------------- semantic communication ----------------
IOTA = 1.1                       # word-to-bit factor iota (T-III)
RHO_TABLE = 0.81                 # rho in T-III; used as the lower bound rho_LB in (34i) (gap 21, accepted)
RHO_LB = RHO_TABLE
RHO_0 = 1.0                      # ASSUMED (gap 21): starting rho in Alg. 3
F_COMP = 10.0                    # coefficient F in Eq. (20) (T-III); no unit given
# Power readings (docs/stage5-power-check.md). Chosen by the project owner on 24 Oct 2026:
# F in mW, Eq. (25) read as energy per task spread over one slot.
F_UNIT = "mW"                    # "mW" or "W"
DT_POWER = "energy_per_slot"     # "energy_per_slot": kappa f^2 C D / dT
                                 # "as_written":      kappa f^3 C        (Eq. 25)
                                 # "kappa_f3":        kappa f^3          (textbook CPU power)

# ---------------- computing ----------------
C_CYCLES_RANGE = (1e3, 2e3)      # CPU cycles per bit (T-III)
D_BITS_RANGE = (1e3, 3e3)        # task data size, bits (T-III)
F_MAX = 5.8e9                    # max RSU CPU frequency, Hz (T-III)
KAPPA = 1e-28                    # CPU energy coefficient (T-III)
NU1 = 243.2e6                    # cycles per degree of RCRB(theta) (T-III; gap 24)
NU2 = 121.6e6                    # cycles per metre of RCRB(d) (T-III; gap 24)
NU3_VAR = 1000.0                 # variance of nu3 (T-III); we use |nu3| (gap 25)

# ---------------- tracking ----------------
Q1_DIAG = (0.02, 0.3, 1.0, 0.1)  # process noise diag(theta, d, v, beta) (T-III); units: gap 14
Q2_DIAG = (0.04, 0.06, 1.0)      # measurement noise diag(sigma1^2, sigma2^2, sigma3^2) (T-III)
NS_PARTICLES = (500, 2000)       # particle counts (Sec. VI-A)

# How the Table III numbers are turned into SI variances. The paper gives no units,
# so these switches make every reading explicit (docs/06-assumptions.md, gaps 14 and 17).
#   angle_unit  "deg2": sigma_theta^2 = 0.02 deg^2 (accepted reading, gap 14)
#               "rad2": sigma_theta^2 = 0.02 rad^2 (sensitivity run)
#   beta_noise  "absolute": u_beta ~ N(0, 0.1), on beta itself, as Eq. (10)-(12) write it
#               "relative": u_beta ~ N(0, 0.1 beta^2), i.e. 32% of beta (ours, not in the paper)
ANGLE_UNIT = "deg2"
BETA_NOISE = "relative"               # chosen by the project owner on 23 Oct 2026 (as-written run kept for the report)
# beta_0 = BETA_SCALE / d. Eq. (10) keeps beta * d constant, so beta ~ 1/d. The constant
# 1/(4 pi) comes from Eq. (6) (Gamma = 1/(sqrt(4 pi) d)); its exact value is ASSUMED (gap 17).
BETA_SCALE = 1.0 / (4 * np.pi)


def q1_si(angle_unit=ANGLE_UNIT):
    """Process noise variances [theta (rad^2), d (m^2), v ((m/s)^2), beta] in SI."""
    th, d, v, b = Q1_DIAG
    if angle_unit == "deg2":
        th = th * (np.pi / 180) ** 2
    elif angle_unit != "rad2":
        raise ValueError(angle_unit)
    return np.array([th, d, v, b])

# ---------------- optimisation ----------------
EPS_WEIGHT = 0.5                 # weight epsilon in Eq. (34a)/(41a) (T-III)
EPS_TOL = 0.1                    # convergence tolerance in Alg. 3 (T-III)
AO_MAX_ITERS = 20                # ASSUMED: cap on Alg. 3 iterations (not given)
AO_SA_ITERS = 2                  # ASSUMED: Alg. 3 iterations while SA scores a candidate (final solve uses AO_MAX_ITERS)
N_RANDOMISATION = 100            # ASSUMED (gap 33): Gaussian randomisation draws
FLIP_PROB = 0.2                  # ASSUMED (gap 30): "greedy with random flip"
SUBSPACE_TOL = 1e-9              # relative singular value cut for the beam subspace (exact up to this)
LBFGS_MAX_ITER = 2000            # ASSUMED: iteration cap of our beamforming solver (final solves)
LBFGS_SA_ITER = 60               # ASSUMED: iteration cap while SA scores candidate assignments (warm-started)
MEG_MAX_ITER = 3000              # cap for the alternative MEG solver
SA_T0 = 100.0                    # SA initial temperature (Alg. 2, line 11)
SA_ALPHA = 0.95                  # ASSUMED (gap 28): standard geometric cooling
SA_TMIN = 1e-3                   # ASSUMED (gap 28)
SA_NMAX = 200                    # ASSUMED (gap 28)

# ---------------- road scenario (Stage 2) ----------------
ROAD_LENGTH = 100.0              # m (Sec. VI)
ROAD_WIDTH = 10.0                # m (Sec. VI)
VEH_LENGTH = 4.5                 # m (Sec. VI)
VEH_WIDTH = 2.0                  # m (Sec. VI)
MIN_GAP = 3.5                    # m (Sec. VI); ASSUMED bumper-to-bumper (gap 6)
LANE_CENTRES_Y = (2.5, 7.5)      # ASSUMED (gap 4): two 5 m lanes
RSU_POSITIONS = ((0.0, -5.0), (100.0, 15.0))  # ASSUMED (gap 1): one RSU each side, 5 m off the road
VEH_SPEED = 20.0                 # m/s, ASSUMED (gap 7)
VEH_HEADING = -1.0               # ASSUMED: vehicles drive towards -x. Eq. (10) has d shrink by
                                 # v dT cos(theta), which is motion along -x for theta measured
                                 # from the +x array axis.
CENTRE_SPACING = VEH_LENGTH + MIN_GAP   # 8 m between centres of cars in the same lane (gap 6)

# ---------------- Stage 1 analysis ----------------
CORR_THRESHOLD = 0.9             # ASSUMED: the paper gives no "material divergence" threshold;
                                 # 0.9 correlation = about 0.9 dB beamforming loss
SEEDS = (0, 1, 2, 3, 4)          # every experiment runs at least these 5 seeds (project rule)
