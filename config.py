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
RHO_TABLE = 0.81                 # rho in T-III; its role is still a "your call" item (gap 21)
F_COMP = 10.0                    # coefficient F in Eq. (20) (T-III); unit unclear, see power check

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

# ---------------- optimisation ----------------
EPS_WEIGHT = 0.5                 # weight epsilon in Eq. (34a) (T-III)
EPS_TOL = 0.1                    # convergence tolerance in Alg. 3 (T-III)
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
