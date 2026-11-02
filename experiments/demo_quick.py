"""QUICK LIVE DEMO (small settings, NOT a paper result).

Runs in about 2 minutes on a 4-core PC. It touches each main model once, so the code can be shown
working in front of an examiner, and it checks the numbers against the committed full runs.

  Part 1  Near vs far field (Stage 1, Eq. 5): Rayleigh distance and how the plane-wave
          model fails close to the array.
  Part 2  Road scenario (Stage 2): one Poisson layout, 3.5 m spacing rule holds.
  Part 3  Tracking (Stage 3, Algorithm 1): K = 5 cars, 5 seeds, PF500, EKF and UKF only
          (PF2000 is left out because it alone takes about 3 minutes per seed).
          Uses exactly the same seeds as experiments/stage3_tracking.py, so the PF500 and EKF
          numbers must match results/stage3_deg2_relative_raw.csv. (UKF is sensitive to
          rounding, so on another computer its numbers can differ a little.)

  Part 4  Assignment + beams (Stage 5, Algorithms 2-3): K = 5, 5 seeds, greedy, HH and
          HH non-semantic, with the full settings (SA_NMAX = 200). Must match
          results/stage5_main_raw.csv, and shows the semantic gain HH vs HH non-semantic.

What is smaller than the full runs: only K = 5 in Parts 3 and 4, no PF2000, and only
three of the five Stage 5 schemes.
Everything else (100 slots, noise values, seeds 0-4) is the same as the full Stage 3 run.

Outputs go to results/demo/ only, so the committed results are never overwritten.

Run from the repository root:  python experiments/demo_quick.py
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one core per worker, as in the README run guide
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import channel   # noqa: E402
import config    # noqa: E402
import scenario  # noqa: E402
from tracking import models  # noqa: E402
from tracking.ekf import EKF  # noqa: E402
from tracking.pf import ParticleFilter  # noqa: E402
from tracking.ukf import UKF  # noqa: E402

OUT = ROOT / "results" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

# Same filter order as stage3_tracking.FILTERS, so each filter gets the same random stream.
FULL_FILTERS = ("PF500", "PF2000", "EKF", "UKF")
DEMO_FILTERS = ("PF500", "EKF", "UKF")


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return np.angle(np.exp(1j * a))


def part1_channel():
    print("=== Part 1: near field vs far field (Eq. 5) ===")
    print(f"Nt = {config.NT} antennas, wavelength = {1e3 * config.WAVELENGTH:.1f} mm")
    print(f"Rayleigh distance = {channel.rayleigh_distance():.1f} m  (paper: about 300 m)")
    theta = np.pi / 2                                        # broadside
    far = channel.far_steering(theta)
    print("Correlation between near-field (Eq. 5) and plane-wave vectors, broadside:")
    for d in (5.0, 20.0, 74.0, 150.0, 300.0):
        c = channel.correlation(channel.nf_steering(theta, d), far)
        print(f"  d = {d:6.1f} m   correlation = {c:.3f}")
    print("Close to the array the plane-wave model is badly wrong; it only becomes good far away.\n")


def part2_road():
    print("=== Part 2: road scenario (K = 10, seed 0) ===")
    lay = scenario.place_vehicles(10, seed=0)
    gap = scenario.min_bumper_gap(lay.x, lay.lane)
    print(f"Car centres x (m): {np.round(np.sort(lay.x), 1)}")
    print(f"Cars per lane: {np.bincount(lay.lane, minlength=2)}")
    print(f"Smallest bumper-to-bumper gap = {gap:.2f} m  (rule: >= {config.MIN_GAP} m)")
    same = scenario.place_vehicles(10, seed=0)
    print(f"Same seed gives same layout: {np.array_equal(lay.x, same.x)}\n")


def track_one(k, seed):
    """PF500, EKF, UKF for one (K, seed). Same loop and seeds as stage3_tracking.run_job."""
    q1 = config.q1_si("deg2")
    beta_noise = "relative"
    lay = scenario.place_vehicles(k, seed)
    power = config.PT_W / k                                  # ASSUMED (gap 16), as in Stage 3
    n = config.N_SLOTS
    sq = {f: np.zeros(3) for f in DEMO_FILTERS}
    t_used = {f: 0.0 for f in DEMO_FILTERS}
    for m in range(config.M_RSU):
        q0 = scenario.initial_states(lay, m)
        for veh in range(k):
            rng_truth = np.random.default_rng([seed, m, veh, 1])
            rng_meas = np.random.default_rng([seed, m, veh, 2])
            rng_init = np.random.default_rng([seed, m, veh, 3])
            truth = models.propagate(q0[veh], n, rng_truth, q1, beta_noise)
            beams = [models.probe_beam(*models.g1(truth[i - 1])[:2], power) for i in range(1, n + 1)]
            meas = [models.measure(truth[i], beams[i - 1], rng_meas) for i in range(1, n + 1)]
            mean0 = q0[veh] + rng_init.standard_normal(4) * np.sqrt(q1)
            for name in DEMO_FILTERS:
                fi = FULL_FILTERS.index(name)
                rng_f = np.random.default_rng([seed, m, veh, 10 + fi])
                if name.startswith("PF"):
                    f = ParticleFilter(mean0, q1, int(name[2:]), rng_f, q1, beta_noise)
                else:
                    f = (EKF if name == "EKF" else UKF)(mean0, q1, q1, beta_noise)
                est = mean0.copy()
                for i in range(1, n + 1):
                    t0 = time.perf_counter()
                    try:
                        new = f.step(meas[i - 1], beams[i - 1])
                        ok = np.all(np.isfinite(new))
                    except np.linalg.LinAlgError:
                        ok = False
                    t_used[name] += time.perf_counter() - t0
                    est = new.copy() if ok else models.g1(est)    # filter broke: coast on g1
                    e = np.array([wrap(est[0] - truth[i, 0]), est[1] - truth[i, 1], est[2] - truth[i, 2]])
                    sq[name] += e**2
    count = config.M_RSU * k * n
    rows = []
    for name in DEMO_FILTERS:
        rmse = np.sqrt(sq[name] / count)
        rows.append(dict(K=k, seed=seed, filter=name, rmse_theta_deg=np.rad2deg(rmse[0]), rmse_d_m=rmse[1],
                         rmse_v_mps=rmse[2], time_per_slot_ms=1e3 * t_used[name] / (config.M_RSU * n)))
    return rows


def part3_tracking(k=5):
    print(f"=== Part 3: tracking, K = {k}, seeds {list(config.SEEDS)}, {config.N_SLOTS} slots ===")
    t0 = time.time()
    rows = []
    for seed in config.SEEDS:
        rows += track_one(k, seed)
        print(f"  seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "demo_tracking_raw.csv", index=False)
    cols = ["rmse_theta_deg", "rmse_d_m", "rmse_v_mps", "time_per_slot_ms"]
    print("\nMean and std over 5 seeds:")
    print(df.groupby("filter")[cols].agg(["mean", "std"]).round(3).to_string())
    print("Paper Fig. 3 (by eye): PF velocity RMSE about 0.8 m/s.")

    # Check against the committed full run (same seeds, so PF500 and EKF should match exactly).
    ref_file = ROOT / "results" / "stage3_deg2_relative_raw.csv"
    if ref_file.exists():
        ref = pd.read_csv(ref_file)
        ref = ref[(ref.K == k) & ref["filter"].isin(DEMO_FILTERS)]
        both = df.merge(ref, on=["K", "seed", "filter"], suffixes=("", "_ref"))
        print("\nLargest difference from the committed full run (results/stage3_deg2_relative_raw.csv):")
        for name in DEMO_FILTERS:
            b = both[both["filter"] == name]
            diff = max(abs(b[c] - b[c + "_ref"]).max() for c in ("rmse_theta_deg", "rmse_d_m", "rmse_v_mps"))
            print(f"  {name:6s} {diff:.2e}")
        print("PF500 and EKF should show about 1e-16 (same numbers). UKF can differ between computers:")
        print("its covariance gets close to singular, so tiny rounding differences change its path.")
        print("Times always differ: they depend on the computer.")
    print(f"Saved {OUT / 'demo_tracking_raw.csv'}\n")


def part4_allocation(k=5, workers=4):
    import stage5_allocation as s5                          # same folder as this script
    print(f"=== Part 4: assignment and beams, K = {k}, seeds {list(config.SEEDS)} ===")
    t0 = time.time()
    schemes = ("greedy", "HH", "HH_nonsem")
    jobs = [(k, seed, sc, config.SA_NMAX) for seed in config.SEEDS for sc in schemes]
    with Pool(workers) as pool:
        rows = [rw for rs in pool.map(s5.job, jobs, chunksize=1) for rw in rs]
    df = pd.DataFrame(rows).sort_values(["scheme", "seed"])
    df.to_csv(OUT / "demo_allocation_raw.csv", index=False)
    cols = ["rate_bps", "rcrb_theta_deg", "rcrb_d_m", "seconds"]
    print("\nMean and std over 5 seeds:")
    print(df.groupby("scheme")[cols].agg(["mean", "std"]).round(4).to_string())
    rate = df.groupby("scheme")["rate_bps"].mean()
    print(f"\nHH beats greedy on rate: {rate['HH'] > rate['greedy']}")
    print(f"Semantic gain, HH over HH non-semantic: {100 * (rate['HH'] / rate['HH_nonsem'] - 1):.1f} %"
          "  (paper claims about 20 %)")
    ref_file = ROOT / "results" / "stage5_main_raw.csv"
    if ref_file.exists():
        ref = pd.read_csv(ref_file)
        both = df.merge(ref, on=["K", "seed", "scheme"], suffixes=("", "_ref"))
        diff = max(abs(both[c] - both[c + "_ref"]).max() for c in ("rate_bps", "rcrb_theta_deg", "rcrb_d_m"))
        print(f"Largest difference from the committed full run (results/stage5_main_raw.csv): {diff:.2e}")
    print(f"Part 4 took {time.time() - t0:.0f} s. Saved {OUT / 'demo_allocation_raw.csv'}\n")


def main():
    t0 = time.time()
    part1_channel()
    part2_road()
    part3_tracking()
    part4_allocation()
    print(f"Demo finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
