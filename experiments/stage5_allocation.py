"""Stage 5: vehicle assignment, beamforming and semantic ratio (Figs. 5-7 setting).

For each K and seed, one slot (the vehicles' starting positions), four schemes:
  greedy        Alg. 2 lines 3-5 (nearest RSU), then Alg. 3
  greedy_flip   greedy, each vehicle flipped with probability 0.2 (gap 30), then Alg. 3
  HH            Alg. 2 (greedy start + simulated annealing), Alg. 3 inside
  HH_nonsem     same, with rho = 1, iota = 1 and no P_Comp (gap 31)
Main runs use latency="relaxed" (L = |nu3|), because with Table III's nu1, nu2 the
latency limit (34g)/(34h) fails for every assignment we tried (docs/stage5-notes.md).
For each final assignment we also report whether it meets (34g)/(34h) as written.

Run from the repository root (single-threaded BLAS is much faster here):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/stage5_allocation.py
Options: --ks 5 10 --seeds 0 1 --workers 4 --nmax 200 --tag main
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # set before numpy loads; small matrices run faster single-threaded
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
SCHEMES = ("greedy", "greedy_flip", "HH", "HH_nonsem")


def setup(k, seed):
    """Slot geometry, tasks and nu3 for one (K, seed)."""
    lay = scenario.place_vehicles(k, seed)
    q = [scenario.initial_states(lay, m) for m in range(config.M_RSU)]
    th = np.array([x[:, 0] for x in q])
    d = np.array([x[:, 1] for x in q])
    be = np.array([x[:, 3] for x in q])
    rng = np.random.default_rng([seed, k, 5])
    c, dbits = allocation.draw_tasks(k, rng)
    nu3 = allocation.draw_nu3(k, rng)
    return bf.build_slot(th, d, be), d, c, dbits, nu3


def run_scheme(slot, d, c, dbits, nu3, scheme, seed, k, n_max):
    """Assignment by the scheme, then the final Alg. 3 solve. Returns a result row."""
    semantic = scheme != "HH_nonsem"
    rng = np.random.default_rng([seed, k, SCHEMES.index(scheme)])
    t0 = time.time()
    sa_steps, sa_solves = 0, 0
    if scheme == "greedy":
        assign = allocation.greedy(d)
    elif scheme == "greedy_flip":
        assign = allocation.random_flip(allocation.greedy(d), rng)
    else:
        warm = {}

        def solve(a):
            nonlocal sa_solves
            sa_solves += 1
            xi = allocation.to_matrix(a)
            r = bf.inner_problem(slot, xi, c, dbits, nu3, semantic=semantic, randomisation=False,
                                 max_iter=config.LBFGS_SA_ITER, start=warm.get("s"), latency="relaxed",
                                 rho0=warm.get("rho"), ao_max=config.AO_SA_ITERS)
            if r is None:
                return None
            warm["s"] = (r["W"], r["R"])
            warm["rho"] = float(np.median(r["rho"][xi > 0]))
            return r["score"]

        assign, _, hist = allocation.hybrid_heuristic(d, solve, rng, n_max=n_max)
        sa_steps = len(hist)
        if assign is None:
            return dict(K=k, seed=seed, scheme=scheme, feasible=False)
    xi = allocation.to_matrix(assign)
    info = {}
    r = bf.inner_problem(slot, xi, c, dbits, nu3, semantic=semantic, rng=rng, latency="relaxed", info=info)
    if r is None:
        return dict(K=k, seed=seed, scheme=scheme, feasible=False, reason=info.get("reason"))
    # Would this assignment meet (34g)/(34h) with the workload L of Eq. (24) as written?
    l_written = allocation.workload(np.rad2deg(np.sqrt(r["crb_theta"])), np.sqrt(r["crb_d"]), nu3)
    f_w, _, ok_w = allocation.computing_plan(xi, c, dbits, l_written)
    rcrb_t = np.rad2deg(np.sqrt(r["crb_theta"]))
    rcrb_d = np.sqrt(r["crb_d"])
    return dict(K=k, seed=seed, scheme=scheme, feasible=True,
                rate_bps=r["rate_bps"].mean(), rate_min=r["rate_bps"].min(),
                rcrb_theta_deg=rcrb_t.mean(), rcrb_theta_deg_median=np.median(rcrb_t),
                rcrb_d_m=rcrb_d.mean(), rcrb_d_m_median=np.median(rcrb_d), rcrb_d_m_max=rcrb_d.max(),
                n_rsu0=int(xi[0].sum()), n_rsu1=int(xi[1].sum()),
                rho_mean=float(r["rho"][xi > 0].mean()),
                p_cs_mw=1e3 * r["p_cs"].sum(), p_comp_mw=1e3 * r["p_comp"].sum(), p_dt_mw=1e3 * r["p_dt"].sum(),
                f_ghz_max=r["f"].sum(axis=1).max() / 1e9,
                written_f_ghz_max=f_w.sum(axis=1).max() / 1e9, written_latency_ok=bool(np.all(ok_w)),
                value_41a=r["value_41a"], ao_iters=r["iterations"],
                sa_steps=sa_steps, sa_solves=sa_solves, seconds=time.time() - t0)


def job(args):
    k, seed, scheme, n_max = args
    slot, d, c, dbits, nu3 = setup(k, seed)
    row = run_scheme(slot, d, c, dbits, nu3, scheme, seed, k, n_max)
    print(f"K={k} seed={seed} {scheme}: " + ", ".join(f"{a}={row[a]:.4g}" for a in ("rate_bps", "rcrb_theta_deg", "rcrb_d_m", "seconds") if a in row),
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=list(config.K_SWEEP))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=config.SA_NMAX)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    jobs = [(k, s, sc, a.nmax) for k in a.ks for s in a.seeds for sc in a.schemes]
    jobs.sort(key=lambda j: (-j[0] * (10 if j[2].startswith("HH") else 1)))       # long jobs first
    with Pool(a.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows).sort_values(["K", "scheme", "seed"])
    out = RESULTS / f"stage5_{a.tag}_raw.csv"
    df.to_csv(out, index=False)
    cols = ["rate_bps", "rcrb_theta_deg", "rcrb_d_m", "rcrb_d_m_median", "written_latency_ok", "seconds"]
    print(df.groupby(["K", "scheme"])[cols].agg(["mean", "std"]).round(4).to_string())
    print("saved", out)


if __name__ == "__main__":
    main()
