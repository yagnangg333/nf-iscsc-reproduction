"""Stage 2: road scenario.

Outputs (results/):
  stage2_layout_example.csv     one layout (K = 10, seed 0) with angle and distance to each RSU
  stage2_layout_stats.csv       spacing statistics per K and seed (5 seeds)
  stage2_acceptance.csv         chance a pure uniform layout meets the 3.5 m rule, per K
  stage2_eq10_drift.csv         noise-free Eq. (10) track vs exact straight-line motion over 2 s
  stage2_layout_example.png

Run from the repository root:  python experiments/stage2_scenario.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config    # noqa: E402
import scenario  # noqa: E402
from tracking import models  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
KS = config.K_SWEEP
BLUE, ORANGE = "#2a78d6", "#eb6834"      # dataviz categorical slots 1 and 2


def nearest_gaps(x, lane):
    """Bumper-to-bumper gap from each car to the next car ahead in its lane."""
    gaps = []
    for ln in np.unique(lane):
        xs = np.sort(x[lane == ln])
        gaps.extend(np.diff(xs) - config.VEH_LENGTH)
    return np.array(gaps)


def layout_stats():
    rows = []
    for k in KS:
        for seed in config.SEEDS:
            lay = scenario.place_vehicles(k, seed)
            g = nearest_gaps(lay.x, lay.lane)
            # Same K placed with no spacing rule (pure Poisson given K), for contrast.
            rng = np.random.default_rng(1000 + seed)
            xu, lu = scenario.naive_uniform(k, rng)
            gu = nearest_gaps(xu, lu)
            rows.append(dict(K=k, seed=seed, lane0=int(np.sum(lay.lane == 0)),
                             density_per_lane_m=k / (2 * config.ROAD_LENGTH),
                             min_gap_m=g.min() if g.size else np.nan,
                             mean_gap_m=g.mean() if g.size else np.nan,
                             min_gap_no_rule_m=gu.min() if gu.size else np.nan))
    return pd.DataFrame(rows)


def eq10_drift():
    """How far a noise-free Eq. (10) track drifts from the true straight-line motion."""
    rows = []
    for k in KS:
        for seed in config.SEEDS:
            lay = scenario.place_vehicles(k, seed)
            for m in range(config.M_RSU):
                q = models.propagate(scenario.initial_states(lay, m), config.N_SLOTS)
                xt, yt = scenario.move(lay, config.N_SLOTS * config.DELTA_T)
                th, d = scenario.to_polar(xt, yt, m)
                dth = np.angle(np.exp(1j * (q[-1, :, 0] - th)))   # wrap to (-pi, pi]
                rows.append(dict(K=k, seed=seed, rsu=m + 1,
                                 max_abs_d_err_m=np.max(np.abs(q[-1, :, 1] - d)),
                                 max_abs_theta_err_deg=np.rad2deg(np.max(np.abs(dth)))))
    return pd.DataFrame(rows)


def plot_example(lay):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.add_patch(Rectangle((0, 0), config.ROAD_LENGTH, config.ROAD_WIDTH, color="0.92", lw=0))
    ax.axhline(config.ROAD_WIDTH / 2, xmin=0.05, xmax=0.95, color="0.6", ls="--", lw=1)
    for xi, yi in zip(lay.x, lay.y):
        ax.add_patch(Rectangle((xi - config.VEH_LENGTH / 2, yi - config.VEH_WIDTH / 2),
                               config.VEH_LENGTH, config.VEH_WIDTH, color="0.35", lw=0))
    for m, col in zip(range(config.M_RSU), (BLUE, ORANGE)):
        rx, ry = config.RSU_POSITIONS[m]
        ax.plot(rx, ry, marker="^", ms=10, color=col, ls="none", label=f"RSU {m + 1}")
        th, d = scenario.to_polar(lay.x, lay.y, m)
        # draw lines to the nearest and farthest car only, to keep it readable
        for i in (np.argmin(d), np.argmax(d)):
            ax.plot([rx, lay.x[i]], [ry, lay.y[i]], color=col, lw=1)
            ax.text((rx + lay.x[i]) / 2, (ry + lay.y[i]) / 2 + 0.6,
                    f"{d[i]:.1f} m, {np.rad2deg(th[i]):.0f} deg", color="0.2", fontsize=7, ha="center")
    ax.annotate("", xy=(80, -3.2), xytext=(95, -3.2), arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.text(87.5, -4.6, f"traffic, {config.VEH_SPEED:.0f} m/s", ha="center", fontsize=7, color="0.3")
    ax.set_xlim(-6, 106)
    ax.set_ylim(-8, 18)
    ax.set_aspect("equal")
    ax.set_xlabel("x along road (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Example layout: K = {lay.x.size}, seed 0 (angles from the +x array axis)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "stage2_layout_example.png", dpi=150)
    plt.close(fig)


def main():
    lay = scenario.place_vehicles(10, 0)
    ex = pd.DataFrame(dict(vehicle=np.arange(1, 11), x_m=lay.x, y_m=lay.y, lane=lay.lane,
                           speed_mps=lay.speed))
    for m in range(config.M_RSU):
        th, d = scenario.to_polar(lay.x, lay.y, m)
        ex[f"theta_rsu{m + 1}_deg"], ex[f"d_rsu{m + 1}_m"] = np.rad2deg(th), d
    ex.to_csv(RESULTS / "stage2_layout_example.csv", index=False)
    plot_example(lay)

    stats = layout_stats()
    stats.to_csv(RESULTS / "stage2_layout_stats.csv", index=False)
    acc = pd.DataFrame(dict(K=list(KS), p_uniform_meets_rule=[scenario.acceptance_probability(k) for k in KS]))
    acc.to_csv(RESULTS / "stage2_acceptance.csv", index=False)
    drift = eq10_drift()
    drift.to_csv(RESULTS / "stage2_eq10_drift.csv", index=False)

    summ = stats.groupby("K").agg(min_gap_mean=("min_gap_m", "mean"), min_gap_std=("min_gap_m", "std"),
                                  mean_gap_mean=("mean_gap_m", "mean"), mean_gap_std=("mean_gap_m", "std"),
                                  no_rule_min_gap_mean=("min_gap_no_rule_m", "mean"))
    print(ex.round(2).to_string(index=False))
    print("\nSpacing over 5 seeds (bumper-to-bumper, same lane):")
    print(summ.round(2).to_string())
    print("\nChance a pure uniform layout already meets the rule:")
    print(acc.to_string(index=False))
    print("\nEq. (10) noise-free drift after 2 s (mean and std over seeds, worst car):")
    print(drift.groupby("rsu")[["max_abs_d_err_m", "max_abs_theta_err_deg"]].agg(["mean", "std", "max"]).round(3))


if __name__ == "__main__":
    main()
