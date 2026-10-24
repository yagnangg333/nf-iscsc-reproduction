"""Stage 6 plot: rate of HH and its baselines at 1 Kb, and the HH gain, ours vs the paper (by eye).

Run from the repository root:  python experiments/stage6_plots.py --tag main
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCHEMES = {"HH": ("Hybrid heuristic", "#eda100"), "HH_nonsem": ("HH non-semantic", "#7a4fc4"),
           "HH_nr1": ("HH with Nr = 1", "#2f9e44"), "benchmark18_nr1": ("Benchmark [18], Nr = 1", "#1fa8c9")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    raw = pd.read_csv(RESULTS / f"stage6_{a.tag}_raw.csv")
    gain = pd.read_csv(RESULTS / f"stage6_{a.tag}_gain.csv")
    paper = pd.concat([pd.read_csv(ROOT / "docs" / "paper_values" / "fig5_7_by_eye.csv"),
                       pd.read_csv(ROOT / "docs" / "paper_values" / "fig5_nr1_by_eye.csv")])
    ok = raw[(raw.feasible == True) & (raw.data_kb == 1.0)]           # noqa: E712
    s = ok.groupby(["K", "scheme"]).rate_bps.agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for sc, (name, col) in SCHEMES.items():
        g = s[s.scheme == sc]
        if len(g):
            ax[0].errorbar(g.K, g["mean"], yerr=g["std"], fmt="o-", color=col, ms=4, capsize=3, label=f"{name}, ours")
        p = paper[paper.scheme == sc]
        ax[0].plot(p.K, p.rate_bps, "x:", color=col, ms=6, label=f"{name}, paper")
    ax[0].set_xlabel("Number of vehicles K")
    ax[0].set_ylabel("Average semantic rate (bps/Hz)")
    ax[0].set_title("D = 1 Kb (10 and 100 Kb: see feasibility CSV)", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False, ncol=2)

    pw = paper.pivot_table(index="K", columns="scheme", values="rate_bps")
    for base, mark in (("HH_nonsem", "o"), ("HH_nr1", "s")):
        name, col = SCHEMES[base]
        g = gain[(gain.baseline == base) & (gain.data_kb == 1.0)]
        ax[1].errorbar(g.K, g.rate_gain_pct_mean, yerr=g.rate_gain_pct_std, fmt=mark + "-", color=col, ms=4,
                       capsize=3, label=f"HH over {name}, ours")
        ax[1].plot(pw.index, 100 * (pw["HH"] / pw[base] - 1), mark + ":", color=col, mfc="none", ms=6,
                   label=f"HH over {name}, paper")
    ax[1].axhline(20, color="0.4", ls="--", lw=1)
    ax[1].text(12.5, 17, "abstract: 20%", ha="center", fontsize=8, color="0.3")
    ax[1].set_xlabel("Number of vehicles K")
    ax[1].set_ylabel("Semantic rate gain of HH (%)")
    ax[1].set_title("D = 1 Kb, gain per seed, mean +/- std", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False, loc="center right")
    for x in ax:
        x.grid(True, color="0.9", lw=0.6)
    fig.tight_layout()
    fig.savefig(RESULTS / f"stage6_{a.tag}_gain.png", dpi=140)
    plt.close(fig)
    print(100 * (pw["HH"] / pw["HH_nonsem"] - 1).round(3), 100 * (pw["HH"] / pw["HH_nr1"] - 1).round(3))


if __name__ == "__main__":
    main()
