"""Road scenario: two-lane highway segment, two RSUs, K vehicles.

From the paper (Sec. II and Sec. VI):
  - straight double-lane road parallel to the RSUs, vehicles at constant
    velocity in the same direction
  - segment 100 m x 10 m, vehicles 4.5 m x 2 m, minimum distance 3.5 m
  - "positions of vehicles follow a Poisson distribution"

ASSUMED (see docs/06-assumptions.md): lane centres (gap 4), RSU positions
(gap 1), fixed K with uniform placement (gap 5), 3.5 m as a bumper-to-bumper
gap within a lane (gap 6), 20 m/s towards -x (gap 7 and config.VEH_HEADING).

Coordinates: x along the road (0 to 100 m), y across it (0 to 10 m).
Vehicle position = centre of its rectangle.

How the spacing rule is enforced (a 1-D hard-core process, exact):
  A Poisson process with a known count K places the K points independently
  and uniformly. We keep that, but only accept layouts where cars in the same
  lane have centres at least s = 4.5 + 3.5 = 8 m apart. Instead of drawing and
  rejecting (slow at K = 20), each lane uses the "shift" trick:
    1. n cars in a lane can use centre positions in [a, b] with b - a = L_c.
    2. Remove the forced gaps: free length L_f = L_c - (n - 1) s.
    3. Draw n uniform points on [0, L_f], sort them, then add 0, s, 2s, ...
  For a fixed lane count this is exactly the uniform distribution over valid
  layouts. Lane counts are drawn with the matching weights (see
  lane_count_weights), so the whole layout has the same law as "place K cars
  uniformly, reject until the rule holds", without the rejections.
"""
from math import comb
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class Layout:
    """Vehicle centres (x, y), lane index and speed for one draw."""
    x: np.ndarray        # (K,) m
    y: np.ndarray        # (K,) m
    lane: np.ndarray     # (K,) 0 or 1
    speed: np.ndarray    # (K,) m/s


def max_per_lane():
    """Most cars that fit in one lane of the segment under the spacing rule."""
    usable = config.ROAD_LENGTH - config.VEH_LENGTH          # range of centre positions
    return int(np.floor(usable / config.CENTRE_SPACING)) + 1


def lane_ok_probability(n):
    """Chance that n uniform cars in one lane already meet the spacing rule.

    Classic 1-D hard-core result: ((U - (n - 1) s) / U)^n, with U the range of
    centre positions and s the centre spacing. Zero if they cannot fit.
    """
    if n <= 1:
        return 1.0
    usable = config.ROAD_LENGTH - config.VEH_LENGTH
    free = usable - (n - 1) * config.CENTRE_SPACING
    return 0.0 if free < 0 else (free / usable) ** n


def lane_count_weights(k):
    """P(n cars in lane 0) for the rejection-sampling law: binomial times acceptance."""
    w = np.array([comb(k, n) * lane_ok_probability(n) * lane_ok_probability(k - n)
                  for n in range(k + 1)])
    return w / w.sum()


def acceptance_probability(k):
    """Chance that a pure uniform (Poisson-given-K) layout meets the rule."""
    return sum(comb(k, n) * 0.5**k * lane_ok_probability(n) * lane_ok_probability(k - n)
               for n in range(k + 1))


def _place_lane(n, rng):
    """Centres of n cars in one lane, uniform over all valid layouts (shift trick)."""
    lo = config.VEH_LENGTH / 2                                 # car fully inside the segment
    usable = config.ROAD_LENGTH - config.VEH_LENGTH
    free = usable - (n - 1) * config.CENTRE_SPACING
    if free < 0:
        raise ValueError(f"{n} cars do not fit in one lane")
    u = np.sort(rng.uniform(0.0, free, n))
    return lo + u + np.arange(n) * config.CENTRE_SPACING


def place_vehicles(k, seed):
    """Place k vehicles on the road. Same seed -> same layout."""
    cap = max_per_lane()
    if k > 2 * cap:
        raise ValueError(f"K = {k} exceeds road capacity {2 * cap}")
    rng = np.random.default_rng(seed)
    # Number of cars in lane 0, with the weights rejection sampling would give.
    n0 = rng.choice(k + 1, p=lane_count_weights(k))
    lane = rng.permutation(np.r_[np.zeros(n0, int), np.ones(k - n0, int)])
    x = np.empty(k)
    for ln in (0, 1):
        idx = np.nonzero(lane == ln)[0]
        # Which car gets which slot is random; positions are already sorted.
        x[rng.permutation(idx)] = _place_lane(idx.size, rng)
    y = np.asarray(config.LANE_CENTRES_Y)[lane]
    speed = np.full(k, config.VEH_SPEED)
    return Layout(x=x, y=y, lane=lane, speed=speed)


def naive_uniform(k, rng):
    """k cars placed uniformly with no spacing rule (pure Poisson given count)."""
    lane = rng.integers(0, 2, k)
    x = rng.uniform(config.VEH_LENGTH / 2, config.ROAD_LENGTH - config.VEH_LENGTH / 2, k)
    return x, lane


def min_bumper_gap(x, lane):
    """Smallest bumper-to-bumper gap between cars in the same lane (inf if none)."""
    gap = np.inf
    for ln in np.unique(lane):
        xs = np.sort(x[lane == ln])
        if xs.size > 1:
            gap = min(gap, np.min(np.diff(xs)) - config.VEH_LENGTH)
    return gap


def move(layout, t):
    """True Cartesian positions after t seconds of constant-velocity motion."""
    x = layout.x + config.VEH_HEADING * layout.speed * t
    return x, layout.y.copy()


def to_polar(x, y, rsu_index):
    """Angle (from the +x array axis, rad) and distance (m) of points seen from an RSU.

    Angles are signed in (-pi, pi]. A ULA cannot tell theta from -theta
    (Eq. 5 uses only cos(theta) and sin^2(theta)), so the sign only matters
    for the motion model.
    """
    rx, ry = config.RSU_POSITIONS[rsu_index]
    dx, dy = np.asarray(x) - rx, np.asarray(y) - ry
    return np.arctan2(dy, dx), np.hypot(dx, dy)


def initial_states(layout, rsu_index):
    """Initial state q = [theta, d, v, beta] per vehicle, as in Eq. (10).

    v is the speed (m/s). beta_0 = BETA_SCALE / d = 1 / (4 pi d), because the
    beta update in Eq. (10) keeps beta * d constant (ASSUMED constant, gap 17).
    Shape (K, 4).
    """
    theta, d = to_polar(layout.x, layout.y, rsu_index)
    beta = config.BETA_SCALE / d
    return np.column_stack([theta, d, layout.speed, beta])
