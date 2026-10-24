"""Stage 5 plots: Figs. 5-7 (ours vs the paper, read by eye) from results/stage5_<tag>_raw.csv.

Run from the repository root:  python experiments/stage5_plots.py --tag main
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCHEMES = {"greedy": ("Greedy", "#2a78d6"), "greedy_flip": ("Greedy with random flip", "#eb6834"),
           "HH": ("Hybrid heuristic", "#eda100"), "HH_nonsem": ("HH non-semantic", "#7a4fc4")}
PANELS = (("rate_bps", "Average semantic rate (bps/Hz)", "Fig. 5", False),
          ("rcrb_theta_deg", "Average RCRB of angle (deg)", "Fig. 6", True),
          ("rcrb_d_m", "Average RCRB of distance (m)", "Fig. 7", True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    raw = pd.read_csv(RESULTS / f"stage5_{a.tag}_raw.csv")
    ok = raw[raw.feasible == True]                                       # noqa: E712
    cols = [p[0] for p in PANELS] + ["rcrb_d_m_median", "rho_mean", "p_dt_mw", "p_comp_mw", "f_ghz_max"]
    summ = ok.groupby(["K", "scheme"])[cols].agg(["mean", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(RESULTS / f"stage5_{a.tag}_summary.csv", index=False)
    paper = pd.read_csv(ROOT / "docs" / "paper_values" / "fig5_7_by_eye.csv")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for (col, ylabel, title, logy), axi in zip(PANELS, ax):
        for sc, (name, colour) in SCHEMES.items():
            s = summ[summ.scheme == sc]
            if len(s):
                axi.errorbar(s.K, s[f"{col}_mean"], yerr=s[f"{col}_std"], fmt="o-", color=colour, ms=4,
                             capsize=3, label=f"{name}, ours")
            p = paper[paper.scheme == sc]
            axi.plot(p.K, p[col], "x:", color=colour, ms=6, label=f"{name}, paper")
        if logy:
            axi.set_yscale("log")
        axi.set_xlabel("Number of vehicles K")
        axi.set_ylabel(ylabel)
        axi.set_title(f"{title} setting (mean +/- std over seeds)", fontsize=9)
        axi.grid(True, color="0.9", lw=0.6)
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / f"stage5_{a.tag}_fig5_7.png", dpi=140)
    plt.close(fig)

    show = summ[["K", "scheme", "rate_bps_mean", "rate_bps_std", "rcrb_theta_deg_mean", "rcrb_d_m_mean",
                 "rcrb_d_m_median_mean", "rate_bps_count"]]
    print(show.round(4).to_string(index=False))
    for k in sorted(summ.K.unique()):
        hh = summ[(summ.K == k) & (summ.scheme == "HH")]
        ns = summ[(summ.K == k) & (summ.scheme == "HH_nonsem")]
        if len(hh) and len(ns):
            drop = 1 - ns.rate_bps_mean.iloc[0] / hh.rate_bps_mean.iloc[0]
            print(f"K={k}: non-semantic rate drop {100 * drop:.1f}% (paper, K = 5: 18%)")
    print("written latency (34g)/(34h) met:", ok.groupby("scheme").written_latency_ok.mean().round(3).to_dict())


if __name__ == "__main__":
    sys.exit(main())
