"""Stage 3: PF (Ns = 500, 2000), EKF and UKF tracking vs number of vehicles (paper Fig. 3).

Set-up (all choices are in config.py or below, and listed in results-log.md):
  - true tracks: Eq. (11) state model = Eq. (10) + process noise u ~ N(0, Q1), run
    separately for every RSU-vehicle pair, as Eq. (10) is written per pair
  - start states from the Stage 2 layout, beta_0 = 1 / (4 pi d)
  - measurements: Eq. (9) echo vector + d_hat + v_hat (tracking/models.py), noise Q2
  - probe beam x for slot i: near-field beam at g1(true state at i-1), power Pt / K.
    The paper steers it at each filter's own prediction; we use one common
    prediction so all four filters see exactly the same measurements.
  - every filter starts from the same draw q0 + N(0, Q1) with covariance Q1
  - RMSE over all vehicles, both RSUs and slots 1..100; time = filter time per slot
    for all K vehicles of one RSU (time.perf_counter, filters only)

Usage (from the repository root):
  python experiments/stage3_tracking.py --angle-unit deg2 --beta-noise absolute
  python experiments/stage3_tracking.py --angle-unit rad2 --beta-noise absolute --ks 5 20
Outputs: results/stage3_<tag>_raw.csv, results/stage3_<tag>_truth.csv, results/stage3_<tag>.png
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config    # noqa: E402
import scenario  # noqa: E402
from tracking import models  # noqa: E402
from tracking.ekf import EKF  # noqa: E402
from tracking.pf import ParticleFilter  # noqa: E402
from tracking.ukf import UKF  # noqa: E402

RESULTS = ROOT / "results"
FILTERS = ("PF500", "PF2000", "EKF", "UKF")


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return np.angle(np.exp(1j * a))


def make_filter(name, mean0, cov0, q1, beta_noise, rng):
    if name.startswith("PF"):
        return ParticleFilter(mean0, cov0, int(name[2:]), rng, q1, beta_noise)
    return (EKF if name == "EKF" else UKF)(mean0, cov0, q1, beta_noise)


def run_job(args):
    """All filters for one (K, seed): returns per-filter error sums, times and truth checks."""
    k, seed, angle_unit, beta_noise = args
    q1 = config.q1_si(angle_unit)
    lay = scenario.place_vehicles(k, seed)
    power = config.PT_W / k                                  # ASSUMED (gap 16)
    n = config.N_SLOTS
    sq = {f: np.zeros(3) for f in FILTERS}                   # summed squared errors (theta, d, v)
    t_used = {f: 0.0 for f in FILTERS}
    fails = {f: 0 for f in FILTERS}
    ess = []
    truth_rows = []
    for m in range(config.M_RSU):
        q0 = scenario.initial_states(lay, m)
        for veh in range(k):
            rng_truth = np.random.default_rng([seed, m, veh, 1])
            rng_meas = np.random.default_rng([seed, m, veh, 2])
            rng_init = np.random.default_rng([seed, m, veh, 3])
            truth = models.propagate(q0[veh], n, rng_truth, q1, beta_noise)
            # exact straight-line track of the same car, for comparison with Eq. (10) + noise
            xe, ye = scenario.move(scenario.Layout(lay.x[veh:veh + 1], lay.y[veh:veh + 1], lay.lane[veh:veh + 1],
                                                   lay.speed[veh:veh + 1]), np.arange(n + 1) * config.DELTA_T)
            th_e, d_e = scenario.to_polar(xe, ye, m)
            truth_rows.append(dict(K=k, seed=seed, rsu=m + 1, vehicle=veh,
                                   rms_theta_diff_deg=np.rad2deg(np.sqrt(np.mean(wrap(truth[:, 0] - th_e) ** 2))),
                                   rms_d_diff_m=np.sqrt(np.mean((truth[:, 1] - d_e) ** 2)),
                                   rms_v_diff_mps=np.sqrt(np.mean((truth[:, 2] - config.VEH_SPEED) ** 2)),
                                   min_d_m=truth[:, 1].min(), min_v_mps=truth[:, 2].min(),
                                   beta_abs_end_over_start=abs(truth[-1, 3]) / abs(truth[0, 3])))
            beams = [models.probe_beam(*models.g1(truth[i - 1])[:2], power) for i in range(1, n + 1)]
            meas = [models.measure(truth[i], beams[i - 1], rng_meas) for i in range(1, n + 1)]
            mean0 = q0[veh] + rng_init.standard_normal(4) * np.sqrt(q1)
            for fi, name in enumerate(FILTERS):
                f = make_filter(name, mean0, q1, q1, beta_noise, np.random.default_rng([seed, m, veh, 10 + fi]))
                est = mean0.copy()
                for i in range(1, n + 1):
                    t0 = time.perf_counter()
                    try:
                        new = f.step(meas[i - 1], beams[i - 1])
                        ok = np.all(np.isfinite(new))
                    except np.linalg.LinAlgError:
                        ok = False
                    t_used[name] += time.perf_counter() - t0
                    if ok:
                        est = new.copy()
                    else:                                   # filter broke: count it, coast on g1
                        fails[name] += 1
                        est = models.g1(est)
                    e = np.array([wrap(est[0] - truth[i, 0]), est[1] - truth[i, 1], est[2] - truth[i, 2]])
                    sq[name] += e**2
                    if name.startswith("PF"):
                        ess.append((name, f.ess))
    count = config.M_RSU * k * n
    rows = []
    for name in FILTERS:
        rmse = np.sqrt(sq[name] / count)
        e = [v for nm, v in ess if nm == name]
        rows.append(dict(K=k, seed=seed, angle_unit=angle_unit, beta_noise=beta_noise, filter=name,
                         rmse_theta_deg=np.rad2deg(rmse[0]), rmse_d_m=rmse[1], rmse_v_mps=rmse[2],
                         time_per_slot_ms=1e3 * t_used[name] / (config.M_RSU * n),
                         failed_steps=fails[name], mean_ess=np.mean(e) if e else np.nan))
    return rows, truth_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle-unit", default=config.ANGLE_UNIT, choices=["deg2", "rad2"])
    ap.add_argument("--beta-noise", default=config.BETA_NOISE, choices=["absolute", "relative"])
    ap.add_argument("--ks", type=int, nargs="+", default=list(config.K_SWEEP))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tag = f"{a.angle_unit}_{a.beta_noise}"
    jobs = [(k, s, a.angle_unit, a.beta_noise) for k in a.ks for s in a.seeds]
    jobs.sort(key=lambda j: -j[0])                           # big jobs first
    t0 = time.time()
    with Pool(a.workers) as pool:
        out = pool.map(run_job, jobs, chunksize=1)
    raw = pd.DataFrame([r for rows, _ in out for r in rows]).sort_values(["K", "seed", "filter"])
    truth = pd.DataFrame([r for _, tr in out for r in tr]).sort_values(["K", "seed", "rsu", "vehicle"])
    RESULTS.mkdir(exist_ok=True)
    raw.to_csv(RESULTS / f"stage3_{tag}_raw.csv", index=False)
    truth.to_csv(RESULTS / f"stage3_{tag}_truth.csv", index=False)
    summ = raw.groupby(["K", "filter"])[["rmse_theta_deg", "rmse_d_m", "rmse_v_mps", "time_per_slot_ms",
                                         "failed_steps", "mean_ess"]].agg(["mean", "std"])
    print(f"config {tag}, {len(jobs)} jobs, {time.time() - t0:.0f} s wall")
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())
    print("\nTruth (Eq. 10 + noise) vs exact straight-line track, mean over vehicles:")
    print(truth.groupby("K")[["rms_theta_diff_deg", "rms_d_diff_m", "rms_v_diff_mps", "min_d_m", "min_v_mps",
                              "beta_abs_end_over_start"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
