"""Stage 6: the [18] benchmark ("Benchmark with Nr = 1" in Fig. 5) and the gain at equal sensing accuracy.

For each (K, seed) and data size that Stage 6 found feasible, we take the final assignment of
"HH with Nr = 1" (same Nr, same vehicles, same tasks) and solve [18] problem (20) + Eq. (21)
(baselines.solve_benchmark18) with P_max = Pt - P_DT, for a sweep of R_min (not given in the paper).
A point counts only if every vehicle's rate after Eq. (21) is at least R_min - RATE_TOL.

Equal sensing accuracy: equal eta = sum of CRB(d) + CRB(theta) over served vehicles, the quantity both
designs minimise. The benchmark's rate at the HH scheme's eta is interpolated over the R_min sweep.
Gain = HH rate / that rate - 1.

Matched comparison (--matched): for each HH-with-Nr = 1 case we re-solve HH's final Alg. 3 on its saved
assignment (fixed rng, so values differ slightly from the Stage 6 run) and give [18] each vehicle's own
Shannon rate from that HH solution as R_min,k. If [18] then reaches an eta no larger than HH's, it gives the
same bits per channel use at equal or better sensing accuracy, and HH's gain over it is the semantic
factor alone: semantic rate / Shannon rate.

Run from the repository root (after stage6_semantic_gain.py):
  OMP_NUM_THREADS=1 python experiments/stage6_benchmark18.py --workers 4 --tag main
  OMP_NUM_THREADS=1 python experiments/stage6_benchmark18.py --workers 4 --tag main --matched
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import allocation  # noqa: E402
import baselines  # noqa: E402
import beamforming as bf  # noqa: E402
import config  # noqa: E402
import stage6_semantic_gain as s6  # noqa: E402

RESULTS = ROOT / "results"
R_MIN_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)   # bit/s/Hz per vehicle; [18] uses 0-10 (Fig. 2) and 5 (Figs. 3-4)
RATE_TOL = 0.05                                      # bit/s/Hz slack for solver accuracy


def job(args):
    k, seed, dbits, assign = args
    warnings.filterwarnings("ignore")
    slot, d, c, dvec, nu3 = s6.setup(k, seed, dbits, 1)
    xi = allocation.to_matrix(np.array([int(ch) for ch in assign]))
    p_avail = config.PT_W - allocation.computing_plan(xi, c, dvec, nu3)[1]
    rows = []
    for r_min in R_MIN_GRID:
        t0 = time.time()
        row = dict(K=k, seed=seed, data_kb=dbits / 1e3, r_min=r_min, valid=False)
        r = baselines.solve_benchmark18_lbfgs(slot, xi, p_avail, r_min)
        if r is None:
            row["status"] = "infeasible or solver failed"
        else:
            rate = bf.rates(slot, xi, r["W"], r["R"], np.ones(xi.shape), 1.0)      # Shannon rate, Eq. (11)
            cd, ct = bf.crbs(slot, xi, r["W"], r["R"])
            row.update(status=r["status"], valid=bool(rate.min() >= r_min - RATE_TOL),
                       rate_bps=rate.mean(), rate_min=rate.min(),
                       rcrb_theta_deg=np.rad2deg(np.sqrt(ct)).mean(), rcrb_d_m=np.sqrt(cd).mean(),
                       eta=float(np.sum(cd + ct)),
                       power_frac=float((bf.beam_power(xi, r["W"], r["R"]) / p_avail).max()),
                       max_violation=r["max_violation"])
        row["seconds"] = time.time() - t0
        rows.append(row)
        print(f"K={k} seed={seed} D={dbits / 1e3:.0f}Kb Rmin={r_min}: " + (
            f"rate={row['rate_bps']:.3f} min={row['rate_min']:.3f} rcrb_d={row['rcrb_d_m']:.3g} valid={row['valid']}"
            if "rate_bps" in row else row["status"]), flush=True)
        if not row["valid"]:
            break
    return rows


def matched_job(args):
    k, seed, dbits, assign = args
    warnings.filterwarnings("ignore")
    slot, d, c, dvec, nu3 = s6.setup(k, seed, dbits, 1)
    xi = allocation.to_matrix(np.array([int(ch) for ch in assign]))
    r = bf.inner_problem(slot, xi, c, dvec, nu3, latency="relaxed", rng=np.random.default_rng([seed, k, 18]))
    row = dict(K=k, seed=seed, data_kb=dbits / 1e3)
    if r is None:
        return dict(row, status="HH re-solve infeasible")
    shannon = bf.rates(slot, xi, r["W"], r["R"], np.ones(xi.shape), 1.0)
    t0 = time.time()
    b = baselines.solve_benchmark18_lbfgs(slot, xi, config.PT_W - r["p_dt"], shannon)
    row.update(hh_rate=r["rate_bps"].mean(), hh_shannon=shannon.mean(), hh_eta=float(np.sum(r["crb_d"] + r["crb_theta"])),
               hh_rho=float(r["rho"][xi > 0].mean()))
    if b is None:
        return dict(row, status="[18] failed")
    sb = bf.rates(slot, xi, b["W"], b["R"], np.ones(xi.shape), 1.0)
    cd, ct = bf.crbs(slot, xi, b["W"], b["R"])
    row.update(status=b["status"], bench_rate=sb.mean(), bench_rate_deficit=float(np.max(shannon - sb)),
               bench_eta=float(np.sum(cd + ct)), seconds=time.time() - t0)
    row["eta_ratio"] = row["bench_eta"] / row["hh_eta"]
    row["gain_pct"] = 100 * (row["hh_rate"] / row["bench_rate"] - 1)
    print(f"K={k} seed={seed} D={dbits / 1e3:.0f}Kb matched: eta ratio {row['eta_ratio']:.4f}, "
          f"gain {row['gain_pct']:.1f}%, deficit {row['bench_rate_deficit']:.3g}", flush=True)
    return row


def equal_accuracy(bench, hh):
    """Per (D, K, seed, scheme): the benchmark's rate at the same eta (sum of CRBs) as the scheme.

    Both designs minimise eta (the CRB part of (34), and [18]'s tr(CRB)), so equal sensing accuracy
    means equal eta. eta rises with R_min, so we interpolate the benchmark's mean rate linearly in
    eta between the two valid grid points on either side of the scheme's eta.
    """
    out = []
    for _, h in hh.iterrows():
        b = bench[(bench.data_kb == h.data_kb) & (bench.K == h.K) & (bench.seed == h.seed) & bench.valid]
        b = b.sort_values("r_min")
        row = dict(data_kb=h.data_kb, K=h.K, seed=h.seed, scheme=h.scheme, hh_rate=h.rate_bps, hh_eta=h.eta)
        if len(b) == 0:
            row["case"] = "no valid benchmark point"
        elif h.eta < b.eta.iloc[0]:
            row["case"] = "benchmark less accurate at every R_min"
        elif h.eta >= b.eta.iloc[-1]:
            row.update(case="benchmark at least this rate (top of grid)", bench_rate=b.rate_bps.iloc[-1])
        else:
            i = int(np.searchsorted(b.eta.values, h.eta))
            e0, e1 = b.eta.iloc[i - 1], b.eta.iloc[i]
            r0, r1 = b.rate_bps.iloc[i - 1], b.rate_bps.iloc[i]
            row.update(case="interpolated", bench_rate=r0 + (r1 - r0) * (h.eta - e0) / max(e1 - e0, 1e-300))
        if "bench_rate" in row:
            row["gain_pct"] = 100 * (h.rate_bps / row["bench_rate"] - 1)
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ks", type=int, nargs="+", default=list(config.K_SWEEP))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    ap.add_argument("--matched", action="store_true")
    a = ap.parse_args()
    raw = pd.read_csv(RESULTS / f"stage6_{a.tag}_raw.csv", dtype={"assign": str})
    ok = raw[(raw.feasible == True) & raw.K.isin(a.ks) & raw.seed.isin(a.seeds)]       # noqa: E712
    src = ok[ok.scheme == "HH_nr1"]
    jobs = [(int(r.K), int(r.seed), r.data_kb * 1e3, r.assign.zfill(int(r.K))) for r in src.itertuples()]
    jobs.sort(key=lambda j: -j[0])
    if a.matched:
        with Pool(a.workers) as pool:
            m = pd.DataFrame(pool.map(matched_job, jobs, chunksize=1)).sort_values(["data_kb", "K", "seed"])
        m.to_csv(RESULTS / f"stage6_{a.tag}_benchmark18_matched.csv", index=False)
        print(m.groupby(["data_kb", "K"])[["eta_ratio", "gain_pct", "hh_rho", "bench_rate_deficit"]]
              .agg(["mean", "std", "count"]).round(4).to_string())
        return
    with Pool(a.workers) as pool:
        bench = pd.DataFrame([row for rows in pool.map(job, jobs, chunksize=1) for row in rows])
    bench = bench.sort_values(["data_kb", "K", "seed", "r_min"])
    bench.to_csv(RESULTS / f"stage6_{a.tag}_benchmark18_raw.csv", index=False)
    eq = equal_accuracy(bench, ok[ok.scheme.isin(["HH", "HH_nonsem", "HH_nr1"])])
    eq.to_csv(RESULTS / f"stage6_{a.tag}_benchmark18_equal_accuracy.csv", index=False)
    v = bench[bench.valid]
    print(v.groupby(["data_kb", "K", "r_min"])[["rate_bps", "rcrb_theta_deg", "rcrb_d_m"]].mean().round(4).to_string())
    print("valid fraction:", bench.groupby("r_min").valid.mean().round(2).to_dict())
    print(eq.groupby(["data_kb", "K", "scheme", "case"]).size().to_string())
    print(eq.groupby(["data_kb", "K", "scheme"]).gain_pct.agg(["mean", "std", "count"]).round(2).to_string())


if __name__ == "__main__":
    main()
