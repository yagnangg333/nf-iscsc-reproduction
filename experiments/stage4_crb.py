"""Stage 4: Cramer-Rao bound for angle (and distance).

Parts
  A1  paper CRB (Eqs. 15-19) vs distance, one vehicle, one beam      -> stage4_crb_vs_distance.csv/.png
  A2  paper CRB vs transmit SNR P / sigma_z^2                         -> stage4_crb_vs_snr.csv
  A3  paper CRB on the road, K = 5..20, equal-power probe beams to all
      K vehicles (Rx = sum_k x_k x_k^H; no optimisation yet)          -> stage4_crb_vs_k.csv
  B   Stage 3 filter RMSE vs one-snapshot CRB and posterior CRB      -> stage4_rmse_vs_bound.csv/.png,
                                                                       stage4_pooled_ratios.csv
  C   Monte Carlo ML vs CRB, 5 seeds                                  -> stage4_ml_check.csv

Paper CRB settings: sigma_z^2 = sigma_r^2 = -30 dBm, T = 1 (ASSUMED, gap 20), beta = 1/(4 pi d).
Part B uses the Stage 3 models and noise (deg2 angles, relative beta noise) so the bound
and the filters see the same problem.

Run from the repository root:  python experiments/stage4_crb.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config    # noqa: E402
import crb       # noqa: E402
import scenario  # noqa: E402
from tracking import models  # noqa: E402

RESULTS = ROOT / "results"
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
ANGLES_DEG = (90.0, 60.0, 30.0)
MC_TRACKS = 10                        # Monte Carlo tracks per vehicle for the posterior CRB


def paper_crb_single(theta, d, power):
    x = models.probe_beam(theta, d, power)[:, None]
    cd, ct = crb.crb_paper(theta, d, config.BETA_SCALE / d, x)
    return np.sqrt(cd), np.rad2deg(np.sqrt(ct))


def part_a():
    rows = []
    for deg in ANGLES_DEG:
        for share, power in (("Pt", config.PT_W), ("Pt/20", config.PT_W / 20)):
            for d in np.geomspace(5, 300, 60):
                rd, rt = paper_crb_single(np.deg2rad(deg), d, power)
                rows.append(dict(theta_deg=deg, power=share, distance_m=d, rcrb_theta_deg=rt, rcrb_d_m=rd))
    dist = pd.DataFrame(rows)
    dist.to_csv(RESULTS / "stage4_crb_vs_distance.csv", index=False)

    rows = []
    for snr_db in np.arange(-10, 61, 5):
        p = config.SIGMA2_R_W * 10 ** (snr_db / 10)
        rd, rt = paper_crb_single(np.deg2rad(60), 50.0, p)
        rows.append(dict(tx_snr_db=snr_db, rcrb_theta_deg=rt, rcrb_d_m=rd))
    snr = pd.DataFrame(rows)
    snr.to_csv(RESULTS / "stage4_crb_vs_snr.csv", index=False)

    rows = []
    for k in config.K_SWEEP:
        for seed in config.SEEDS:
            lay = scenario.place_vehicles(k, seed)
            for m in range(config.M_RSU):
                th, d = scenario.to_polar(lay.x, lay.y, m)
                X = np.column_stack([models.probe_beam(t, dd, config.PT_W / k) for t, dd in zip(th, d)])
                for t, dd in zip(th, d):
                    cd, ct = crb.crb_paper(t, dd, config.BETA_SCALE / dd, X)
                    rows.append(dict(K=k, seed=seed, rsu=m + 1, rcrb_theta_deg=np.rad2deg(np.sqrt(ct)),
                                     rcrb_d_m=np.sqrt(cd)))
    road = pd.DataFrame(rows)
    road.to_csv(RESULTS / "stage4_crb_vs_k.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for deg, col in zip(ANGLES_DEG, COLOURS):
        g = dist[(dist.theta_deg == deg) & (dist.power == "Pt")]
        axes[0].loglog(g.distance_m, g.rcrb_theta_deg, color=col, lw=2, label=f"theta = {deg:.0f} deg")
        axes[1].loglog(g.distance_m, g.rcrb_d_m, color=col, lw=2, label=f"theta = {deg:.0f} deg")
    for ax, lab in zip(axes, ("Root CRB of angle (deg)", "Root CRB of distance (m)")):
        ax.axvline(channel_rayleigh(), color="0.4", ls="--", lw=1)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(lab)
        ax.grid(True, which="both", color="0.9", lw=0.6)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Paper CRB (Eqs. 15-19), one vehicle, beam power Pt, sigma_z^2 = -30 dBm, T = 1; dashed: Rayleigh distance",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "stage4_crb_vs_distance.png", dpi=140)
    plt.close(fig)
    return snr, road


def channel_rayleigh():
    import channel
    return channel.rayleigh_distance()


def bound_job(args):
    """Posterior and snapshot CRB for every vehicle of one (K, seed), Stage 3 set-up."""
    k, seed = args
    q1 = config.q1_si("deg2")
    lay = scenario.place_vehicles(k, seed)
    power = config.PT_W / k
    n = config.N_SLOTS
    sums = np.zeros(3)          # summed PCRB over vehicles and slots: theta, d, v
    snap = []                   # one-snapshot CRB of every slot: theta, d, v
    for m in range(config.M_RSU):
        q0 = scenario.initial_states(lay, m)
        for veh in range(k):
            tracks = np.stack([models.propagate(q0[veh], n, np.random.default_rng([seed, m, veh, 100 + j]), q1, "relative")
                               for j in range(MC_TRACKS)])
            beams = np.array([[models.probe_beam(*models.g1(tr[i - 1])[:2], power) for i in range(1, n + 1)]
                              for tr in tracks])
            p = crb.posterior_crb(tracks, beams, q1, q1, beta_noise="relative")
            sums += p[:, :3].sum(axis=0)
            snap += [crb.snapshot_crb(tr[i], bm[i - 1])[:3] for tr, bm in zip(tracks, beams) for i in range(1, n + 1)]
    count = config.M_RSU * k * n
    snap = np.array(snap)
    # The one-snapshot angle CRB has a very long tail (vehicles near the array axis, where
    # sin(theta) is small), so its mean is meaningless; we report the median for angle.
    return dict(K=k, seed=seed,
                rpcrb_theta_deg=np.rad2deg(np.sqrt(sums[0] / count)), rpcrb_d_m=np.sqrt(sums[1] / count),
                rpcrb_v_mps=np.sqrt(sums[2] / count),
                rsnap_theta_deg_median=np.rad2deg(np.sqrt(np.median(snap[:, 0]))),
                rsnap_d_m=np.sqrt(snap[:, 1].mean()), rsnap_v_mps=np.sqrt(snap[:, 2].mean()))


def part_b():
    jobs = sorted([(k, s) for k in config.K_SWEEP for s in config.SEEDS], key=lambda j: -j[0])
    with Pool(4) as pool:
        bounds = pd.DataFrame(pool.map(bound_job, jobs, chunksize=1))
    raw = pd.read_csv(RESULTS / "stage3_deg2_relative_raw.csv")
    merged = raw.merge(bounds, on=["K", "seed"])
    merged["theta_rmse_over_rpcrb"] = merged.rmse_theta_deg / merged.rpcrb_theta_deg
    merged["d_rmse_over_rpcrb"] = merged.rmse_d_m / merged.rpcrb_d_m
    merged["v_rmse_over_rpcrb"] = merged.rmse_v_mps / merged.rpcrb_v_mps
    merged.to_csv(RESULTS / "stage4_rmse_vs_bound.csv", index=False)

    # Pool the seeds: sqrt(mean RMSE^2 / mean PCRB). One run has only 2K tracks, so its RMSE
    # alone can fall a few % under the bound by chance (docs/stage4-notes.md).
    rows = []
    for (k, f), g in merged.groupby(["K", "filter"]):
        rows.append(dict(K=k, filter=f, **{a: np.sqrt((g["rmse_" + a] ** 2).mean() / (g["rpcrb_" + a] ** 2).mean())
                                           for a in ("theta_deg", "d_m", "v_mps")}))
    pooled = pd.DataFrame(rows)
    pooled.to_csv(RESULTS / "stage4_pooled_ratios.csv", index=False)

    ks = sorted(merged.K.unique())
    filters = ("PF500", "PF2000", "EKF", "UKF")
    panels = (("theta_deg", "Angle error (deg)"), ("d_m", "Distance error (m)"), ("v_mps", "Velocity error (m/s)"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (a, label) in zip(axes, panels):
        for j, (f, col) in enumerate(zip(filters, COLOURS)):
            g = merged[merged["filter"] == f].groupby("K")["rmse_" + a]
            ax.bar(np.arange(len(ks)) + (j - 1.5) * 0.18, g.mean(), 0.16, yerr=g.std(), color=col, label=f,
                   error_kw=dict(lw=0.8, ecolor="0.3"))
        ax.plot(np.arange(len(ks)), bounds.groupby("K")["rpcrb_" + a].mean(), "k_", ms=36, mew=2,
                label="root posterior CRB")
        ax.set_xticks(np.arange(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("Number of vehicles K")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Stage 3 filter RMSE (mean +/- std over 5 seeds) vs root posterior CRB", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(RESULTS / "stage4_rmse_vs_bound.png", dpi=140)
    plt.close(fig)
    return merged, bounds, pooled


def _echo(theta, d, x):
    return crb.b_products(theta, d, x[:, None])[0][:, 0]


def part_c():
    """ML vs CRB at high SNR, 5 seeds x 200 trials (same cases as tests/test_crb.py)."""
    th, d = np.deg2rad(60.0), 30.0
    x = models.probe_beam(th, d, config.PT_W)
    X = x[:, None]
    beta = config.BETA_SCALE / d * np.exp(0.7j)
    rows = []
    for seed in config.SEEDS:
        rng = np.random.default_rng(seed)
        # (i) theta only, d and beta known
        s2 = 1e-3
        ct1 = 1.0 / crb.fim_paper(th, d, beta, X, s2)[1, 1]
        clean = beta * _echo(th, d, x)
        e1 = []
        for _ in range(200):
            z = clean + np.sqrt(s2 / 2) * (rng.standard_normal(config.NT) + 1j * rng.standard_normal(config.NT))
            w = 30 * np.sqrt(ct1)
            e1.append(minimize_scalar(lambda t: np.sum(np.abs(z - beta * _echo(t, d, x)) ** 2),
                                      bounds=(th - w, th + w), method="bounded", options=dict(xatol=1e-12)).x - th)
        # (ii) theta, d, complex beta unknown
        s2 = 1e-4
        cd, ct = crb.crb_paper(th, d, beta, X, s2)
        sc = np.array([np.sqrt(ct), np.sqrt(cd)])
        clean = beta * _echo(th, d, x)
        e2 = []
        for _ in range(200):
            z = clean + np.sqrt(s2 / 2) * (rng.standard_normal(config.NT) + 1j * rng.standard_normal(config.NT))

            def cost(p):
                e = _echo(th + p[0] * sc[0], d + p[1] * sc[1], x)
                return -np.abs(np.vdot(e, z)) ** 2 / np.vdot(e, e).real
            e2.append(minimize(cost, [0.0, 0.0], method="Nelder-Mead",
                               options=dict(xatol=1e-4, fatol=1e-12, maxiter=400)).x * sc)
        e2 = np.array(e2)
        rows.append(dict(seed=seed, theta_only_mse_over_crb=np.mean(np.square(e1)) / ct1,
                         joint_theta_mse_over_crb=np.mean(e2[:, 0] ** 2) / ct,
                         joint_d_mse_over_crb=np.mean(e2[:, 1] ** 2) / cd))
    ml = pd.DataFrame(rows)
    ml.to_csv(RESULTS / "stage4_ml_check.csv", index=False)
    return ml


def main():
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    snr, road = part_a()
    print("A2: paper RCRB vs transmit SNR (d = 50 m, theta = 60 deg):")
    print(snr.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nA3: paper RCRB on the road, equal-power probe beams (mean and std over 5 seeds of the per-seed mean):")
    per_seed = road.groupby(["K", "seed"])[["rcrb_theta_deg", "rcrb_d_m"]].mean()
    print(per_seed.groupby("K").agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.3g}"))
    ml = part_c()
    print("\nC: ML / CRB (mean, std over 5 seeds):")
    print(ml.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())
    merged, bounds, pooled = part_b()
    print("\nB: angle RMSE / root posterior CRB (mean, min over seeds):")
    print(merged.groupby(["K", "filter"]).theta_rmse_over_rpcrb.agg(["mean", "min"]).round(3).to_string())
    print("\nB: distance and velocity RMSE / root posterior CRB, smallest single run:",
          round(merged.d_rmse_over_rpcrb.min(), 3), round(merged.v_rmse_over_rpcrb.min(), 3))
    print("\nB: RMSE / root posterior CRB with the 5 seeds pooled:")
    print(pooled.round(3).to_string(index=False))
    print("\nB: bounds (mean over seeds):")
    print(bounds.groupby("K").mean().drop(columns="seed").round(4).to_string())


if __name__ == "__main__":
    main()
