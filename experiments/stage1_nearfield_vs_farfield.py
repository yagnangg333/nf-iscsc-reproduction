"""Stage 1: near-field vs far-field steering vectors.

Model: Eq. (5) of Yang et al. (Fresnel near-field steering), plus the exact
spherical phase and the plane-wave model (ours) for comparison.

Outputs (results/):
  stage1_corr_vs_distance.csv   correlation and gain loss vs distance, per angle
  stage1_divergence.csv         distance where correlation first drops below 0.9, per angle
  stage1_random_angles.csv      same, for random angles, per seed (5 seeds)
  stage1_road_points.csv        share of road points where a far-field beam loses > 0.9 dB, per seed
  stage1_corr_vs_distance.png, stage1_gain_loss.png

Run from the repository root:  python experiments/stage1_nearfield_vs_farfield.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import channel  # noqa: E402
import config   # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# theta is measured from the array axis, so 90 deg is broadside.
# 90, 60, 30 deg from the axis = 0, 30, 60 deg from broadside.
ANGLES_DEG = (90.0, 60.0, 30.0)
DIST = np.logspace(0, np.log10(500), 1500)        # 1 m to 500 m
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")        # categorical slots 1-3 (dataviz palette)


def corr_curve(theta, model):
    """Correlation between the near-field vector (given model) and the far-field one."""
    far = channel.far_steering(theta)
    f = channel.exact_steering if model == "exact" else channel.nf_steering
    return np.array([channel.correlation(f(theta, d), far) for d in DIST])


def divergence_distance(corr, threshold=config.CORR_THRESHOLD):
    """Largest distance at which correlation is still below the threshold.

    Beyond this distance the far-field model is good enough (corr >= threshold).
    Returns 0 if correlation never drops below the threshold.
    """
    below = np.nonzero(corr < threshold)[0]
    return 0.0 if below.size == 0 else DIST[below[-1]]


def main():
    rows, div_rows, curves = [], [], {}
    for deg in ANGLES_DEG:
        th = np.deg2rad(deg)
        c_exact, c_fres = corr_curve(th, "exact"), corr_curve(th, "fresnel")
        curves[deg] = c_exact
        for d, ce, cf in zip(DIST, c_exact, c_fres):
            rows.append(dict(theta_deg=deg, distance_m=d, corr_exact=ce, corr_eq5=cf,
                             gain_loss_db_exact=20 * np.log10(ce)))
        div_rows.append(dict(theta_deg=deg, threshold=config.CORR_THRESHOLD,
                             divergence_m_exact=divergence_distance(c_exact),
                             divergence_m_eq5=divergence_distance(c_fres)))
    pd.DataFrame(rows).to_csv(RESULTS / "stage1_corr_vs_distance.csv", index=False)
    div = pd.DataFrame(div_rows)
    div.to_csv(RESULTS / "stage1_divergence.csv", index=False)

    # Random angles, 5 seeds: how the divergence distance spreads over directions.
    rand_rows = []
    for seed in config.SEEDS:
        rng = np.random.default_rng(seed)
        thetas = rng.uniform(np.deg2rad(20), np.deg2rad(160), 50)
        dd = [divergence_distance(corr_curve(t, "exact")) for t in thetas]
        rand_rows.append(dict(seed=seed, n_angles=len(dd), mean_divergence_m=np.mean(dd),
                              max_divergence_m=np.max(dd)))
    rand = pd.DataFrame(rand_rows)
    rand.to_csv(RESULTS / "stage1_random_angles.csv", index=False)

    # Road check (preliminary, Stage 2 builds the real scenario): random points on
    # the 100 m x 10 m road. RSU 1 at its ASSUMED position (road end), plus a
    # mid-road position as a sensitivity check, since the paper gives neither.
    placements = {"assumed_end_(0,-5)": (0.0, -5.0), "sensitivity_mid_(50,-5)": (50.0, -5.0)}
    road_rows = []
    for label, pos in placements.items():
        rsu = np.array(pos)
        for seed in config.SEEDS:
            rng = np.random.default_rng(seed)
            x = rng.uniform(0, config.ROAD_LENGTH, 200)
            y = rng.choice(config.LANE_CENTRES_Y, 200)
            dx, dy = x - rsu[0], y - rsu[1]
            dist, theta = np.hypot(dx, dy), np.arctan2(dy, dx)
            corr = np.array([channel.correlation(channel.exact_steering(t, d), channel.far_steering(t))
                             for t, d in zip(theta, dist)])
            road_rows.append(dict(rsu_placement=label, seed=seed, n_points=len(corr),
                                  share_below_threshold=np.mean(corr < config.CORR_THRESHOLD),
                                  mean_gain_loss_db=np.mean(20 * np.log10(corr))))
    road = pd.DataFrame(road_rows)
    road.to_csv(RESULTS / "stage1_road_points.csv", index=False)

    # --- plots ---
    r_ray = channel.rayleigh_distance()
    for name, ylab, transform in (
        ("stage1_corr_vs_distance.png", "|a_near^H a_far| / Nt", lambda c: c),
        ("stage1_gain_loss.png", "Far-field beam gain loss (dB)", lambda c: 20 * np.log10(c)),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for deg, col in zip(ANGLES_DEG, COLOURS):
            ax.semilogx(DIST, transform(curves[deg]), color=col, lw=2,
                        label=f"theta = {deg:.0f} deg ({90 - deg:.0f} deg off broadside)")
        ax.axvline(r_ray, color="0.4", lw=1, ls="--")
        ax.text(r_ray * 0.97, ax.get_ylim()[1], f"Rayleigh {r_ray:.0f} m ", rotation=90,
                va="top", ha="right", color="0.3", fontsize=8)
        ax.axvspan(5, 101, color="0.85", alpha=0.5, lw=0)
        ax.text(7, ax.get_ylim()[1], "road distances\n(RSU 5 m off road)", va="top",
                fontsize=8, color="0.3")
        if name.startswith("stage1_corr"):
            ax.axhline(config.CORR_THRESHOLD, color="0.4", lw=1, ls=":")
        ax.set_xlabel("Distance from array centre (m)")
        ax.set_ylabel(ylab)
        ax.set_title(f"Exact spherical vs plane-wave steering, Nt = {config.NT}, 50 GHz")
        ax.grid(True, which="both", color="0.9", lw=0.6)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=False)
        fig.tight_layout()
        fig.savefig(RESULTS / name, dpi=150)
        plt.close(fig)

    print(div.to_string(index=False))
    print("\nRandom angles (20-160 deg), divergence distance over 5 seeds: "
          f"mean {rand.mean_divergence_m.mean():.1f} m, std {rand.mean_divergence_m.std(ddof=1):.1f} m")
    for label, g in road.groupby("rsu_placement"):
        print(f"Road points with far-field correlation < 0.9, RSU {label}: "
              f"mean {100 * g.share_below_threshold.mean():.1f} %, "
              f"std {100 * g.share_below_threshold.std(ddof=1):.1f} %; "
              f"mean gain loss {g.mean_gain_loss_db.mean():.2f} dB")
    print(f"Rayleigh distance: {r_ray:.1f} m")


if __name__ == "__main__":
    main()
