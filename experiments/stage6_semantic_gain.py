"""Stage 6: the "20% semantic rate gain at equal sensing accuracy" claim, at 1, 10 and 100 Kb.

Same slot and Alg. 2 + Alg. 3 as Stage 5, but every vehicle's task has the same data size D
(1, 10 or 100 Kb; C still drawn from Table III). Schemes:
  HH            proposed (semantic), Nr = 3
  HH_nonsem     HH with semantics off (rho = 1, iota = 1, no P_Comp; gap 31)
  HH_nr1        proposed with Nr = 1 (MISO), as "Hybrid Heuristic with Nr = 1" in Fig. 5
The [18] benchmark ("Benchmark with Nr = 1") is added once we have the [18] paper.
Latency uses L = |nu3| as in Stage 5 (L of Eq. (24) as written is never feasible); F^max still applies.
The paper does not run Fig. 5 at these data sizes: this sweep is our extension (docs/stage0-notes.md).

Run from the repository root:
  OMP_NUM_THREADS=1 python experiments/stage6_semantic_gain.py --workers 4 --tag main
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import allocation  # noqa: E402
import beamforming as bf  # noqa: E402
import config  # noqa: E402
import scenario  # noqa: E402

RESULTS = ROOT / "results"
SCHEMES = {"HH": (True, config.NR), "HH_nonsem": (False, config.NR), "HH_nr1": (True, 1)}
SIZES = (1e3, 1e4, 1e5)                    # bits: 1, 10, 100 Kb


def setup(k, seed, dbits, nr):
    """Slot for (K, seed) with Nr receive antennas; C and nu3 drawn as in Stage 5, D fixed."""
    lay = scenario.place_vehicles(k, seed)
    q = [scenario.initial_states(lay, m) for m in range(config.M_RSU)]
    th, d, be = (np.array([x[:, j] for x in q]) for j in (0, 1, 3))
    rng = np.random.default_rng([seed, k, 5])
    c, _ = allocation.draw_tasks(k, rng)   # same C and nu3 as Stage 5 for this (K, seed)
    nu3 = allocation.draw_nu3(k, rng)
    return bf.build_slot(th, d, be, nr=nr), d, c, np.full(k, dbits), nu3


def run(k, seed, dbits, scheme, n_max):
    semantic, nr = SCHEMES[scheme]
    slot, d, c, dvec, nu3 = setup(k, seed, dbits, nr)
    rng = np.random.default_rng([seed, k, int(dbits), list(SCHEMES).index(scheme)])
    t0 = time.time()
    row = dict(K=k, seed=seed, data_kb=dbits / 1e3, scheme=scheme, feasible=False)
    # Can any assignment meet (34h)? Each vehicle alone needs f = (C D + |nu3|) / T^max.
    f_one = allocation.min_frequency(c, dvec, nu3)
    row["f_one_vehicle_ghz_max"] = f_one.max() / 1e9
    if f_one.max() > config.F_MAX:
        row["reason"] = "cpu (one vehicle needs more than F^max)"
        return row
    warm = {}

    def solve(a):
        xi = allocation.to_matrix(a)
        r = bf.inner_problem(slot, xi, c, dvec, nu3, semantic=semantic, randomisation=False,
                             max_iter=config.LBFGS_SA_ITER, start=warm.get("s"), latency="relaxed",
                             rho0=warm.get("rho"), ao_max=config.AO_SA_ITERS)
        if r is None:
            return None
        warm["s"] = (r["W"], r["R"])
        warm["rho"] = float(np.median(r["rho"][xi > 0]))
        return r["score"]

    assign, _, hist = allocation.hybrid_heuristic(d, solve, rng, n_max=n_max)
    if assign is None:
        row["reason"] = "cpu (no feasible assignment found)"
        return row
    xi = allocation.to_matrix(assign)
    info = {}
    r = bf.inner_problem(slot, xi, c, dvec, nu3, semantic=semantic, rng=rng, latency="relaxed", info=info)
    if r is None:
        row["reason"] = info.get("reason")
        return row
    rcrb_t = np.rad2deg(np.sqrt(r["crb_theta"]))
    rcrb_d = np.sqrt(r["crb_d"])
    row.update(feasible=True, rate_bps=r["rate_bps"].mean(), rcrb_theta_deg=rcrb_t.mean(),
               eta=float(np.sum(r["crb_d"] + r["crb_theta"])),               # sum of CRBs (m^2 + rad^2), as in (34)
               rcrb_d_m=rcrb_d.mean(), rcrb_d_m_median=np.median(rcrb_d),
               rho_mean=float(r["rho"][xi > 0].mean()), p_comp_mw=1e3 * r["p_comp"].sum(),
               p_dt_mw=1e3 * r["p_dt"].sum(), f_ghz_max=r["f"].sum(axis=1).max() / 1e9,
               n_rsu0=int(xi[0].sum()), assign="".join(str(int(v)) for v in assign),
               sa_steps=len(hist), seconds=time.time() - t0)
    return row


def job(args):
    row = run(*args)
    print(f"K={row['K']} seed={row['seed']} D={row['data_kb']:.0f}Kb {row['scheme']}: "
          + (f"rate={row['rate_bps']:.4g}, rcrb_d={row['rcrb_d_m']:.4g}, s={row['seconds']:.0f}"
             if row["feasible"] else f"infeasible ({row.get('reason')})"), flush=True)
    return row


def summarise(df):
    """Gain of HH over each baseline per (D, K): mean and std over seeds of the per-seed ratio."""
    ok = df[df.feasible == True]                                       # noqa: E712
    wide = ok.pivot_table(index=["data_kb", "K", "seed"], columns="scheme",
                          values=["rate_bps", "rcrb_theta_deg", "rcrb_d_m"])
    out = []
    for base in [s for s in SCHEMES if s != "HH"]:
        if ("rate_bps", base) not in wide.columns:
            continue
        g = pd.DataFrame({"rate_gain_pct": 100 * (wide["rate_bps"]["HH"] / wide["rate_bps"][base] - 1),
                          "rcrb_theta_ratio": wide["rcrb_theta_deg"]["HH"] / wide["rcrb_theta_deg"][base],
                          "rcrb_d_ratio": wide["rcrb_d_m"]["HH"] / wide["rcrb_d_m"][base]}).dropna()
        s = g.groupby(["data_kb", "K"]).agg(["mean", "std", "count"])
        s.columns = ["_".join(c) for c in s.columns]
        out.append(s.reset_index().assign(baseline=base))
    feas = df.groupby(["data_kb", "K", "scheme"]).feasible.mean().rename("feasible_fraction").reset_index()
    return (pd.concat(out) if out else pd.DataFrame()), feas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=list(config.K_SWEEP))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    ap.add_argument("--sizes", type=float, nargs="+", default=list(SIZES))
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=config.SA_NMAX)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    jobs = [(k, s, dbits, sc, a.nmax) for dbits in a.sizes for k in a.ks for s in a.seeds for sc in a.schemes]
    jobs.sort(key=lambda j: -j[0])                                     # long jobs first
    with Pool(a.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows).sort_values(["data_kb", "K", "scheme", "seed"])
    df.to_csv(RESULTS / f"stage6_{a.tag}_raw.csv", index=False)
    gains, feas = summarise(df)
    gains.to_csv(RESULTS / f"stage6_{a.tag}_gain.csv", index=False)
    feas.to_csv(RESULTS / f"stage6_{a.tag}_feasibility.csv", index=False)
    print(feas.to_string(index=False))
    print(gains.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
