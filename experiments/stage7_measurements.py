"""Stage 7: measurement tables for every scheme (our list; the comparison framework PDF was not provided).

Reads results/stage5_<tag>_raw.csv (greedy, greedy with flip, HH, HH non-semantic, HH with Nr = 1, [18])
and results/stage6_main_feasibility.csv. Writes, for K = 5-20 and 5 seeds (mean and std):
  results/stage7_measurements.csv   long table: K, scheme, measure, mean, std, n
  results/stage7_measurements.md    one Markdown table per measure (for the report)
  results/stage7_measurements.png   rate, RCRBs, power split and run time vs K

Run from the repository root:  python experiments/stage7_measurements.py --tag stage7
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCHEMES = {"greedy": ("Greedy", "#2a78d6"), "greedy_flip": ("Greedy with flip", "#eb6834"),
           "HH": ("HH", "#eda100"), "HH_nonsem": ("HH non-semantic", "#7a4fc4"),
           "HH_nr1": ("HH, Nr = 1", "#2f9e44"), "bench18": ("[18], Nr = 1", "#1fa8c9")}
MEASURES = {                                    # column: (label, unit)
    "rate_bps": ("Semantic rate, Eq. (13)", "bps/Hz"),
    "shannon_bps": ("Shannon rate (iota = rho = 1)", "bps/Hz"),
    "rcrb_theta_deg": ("Angle RCRB, mean over vehicles", "deg"),
    "rcrb_d_m": ("Distance RCRB, mean over vehicles", "m"),
    "rcrb_d_m_median": ("Distance RCRB, median over vehicles", "m"),
    "eta": ("Sum of CRBs (eta in (34))", "m^2 + rad^2"),
    "f_ghz_max": ("CPU frequency used, busier RSU", "GHz"),
    "p_cs_mw": ("Beam power P_C&S, both RSUs", "mW"),
    "p_comp_mw": ("Semantic extraction power P_Comp, both RSUs", "mW"),
    "p_dt_mw": ("DT power P_DT, both RSUs", "mW"),
    "rho_mean": ("Semantic ratio rho, mean", "-"),
    "sa_seconds": ("Run time: assignment (Alg. 2)", "s"),
    "final_seconds": ("Run time: final beamforming (Alg. 3, or [18])", "s"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage7")
    a = ap.parse_args()
    raw = pd.read_csv(RESULTS / f"stage5_{a.tag}_raw.csv")
    ok = raw[raw.feasible == True]                                      # noqa: E712
    long = []
    for col in MEASURES:
        g = ok.groupby(["K", "scheme"])[col].agg(["mean", "std", "count"]).reset_index()
        g.insert(2, "measure", col)
        long.append(g)
    long = pd.concat(long).rename(columns={"count": "n"})
    long.to_csv(RESULTS / "stage7_measurements.csv", index=False)

    lines = ["# Stage 7 measurements", "",
             f"Source: results/stage5_{a.tag}_raw.csv. Mean +/- std over 5 seeds, one slot per seed. "
             "Latency uses L = |nu3| (docs/stage5-notes.md). [18] runs on the HH, Nr = 1 assignment with each "
             "vehicle's HH, Nr = 1 Shannon rate as R_min (docs/stage6-notes.md). Run times are single-core.", ""]
    for col, (label, unit) in MEASURES.items():
        t = long[long.measure == col]
        lines += [f"## {label} ({unit})", "", "| K | " + " | ".join(SCHEMES[s][0] for s in SCHEMES) + " |",
                  "|---|" + "---|" * len(SCHEMES)]
        for k in sorted(t.K.unique()):
            cells = []
            for s in SCHEMES:
                r = t[(t.K == k) & (t.scheme == s)]
                cells.append("-" if r.empty or pd.isna(r["mean"].iloc[0]) else
                             f"{r['mean'].iloc[0]:.3g} +/- {r['std'].iloc[0]:.2g}")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    feas = pd.read_csv(RESULTS / "stage6_main_feasibility.csv")
    fp = feas.pivot_table(index=["data_kb", "K"], columns="scheme", values="feasible_fraction").reset_index()
    cols = [c for c in fp.columns if c not in ("data_kb", "K")]
    lines += ["## Feasibility at fixed data sizes (fraction of 5 seeds, Stage 6)", "",
              "| D (Kb) | K | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    lines += [f"| {r.data_kb:g} | {r.K} | " + " | ".join(f"{getattr(r, c):.1f}" for c in cols) + " |"
              for r in fp.itertuples()]
    lines.append("")
    (RESULTS / "stage7_measurements.md").write_text("\n".join(lines))

    panels = ("rate_bps", "rcrb_theta_deg", "rcrb_d_m", "p_cs_mw", "p_dt_mw", "final_seconds")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for col, axi in zip(panels, ax.ravel()):
        t = long[long.measure == col]
        for s, (name, colour) in SCHEMES.items():
            r = t[t.scheme == s]
            if len(r):
                axi.errorbar(r.K, r["mean"], yerr=r["std"], fmt="o-", color=colour, ms=4, capsize=3, label=name)
        if col.startswith("rcrb") or col.endswith("seconds"):
            axi.set_yscale("log")
        axi.set_xlabel("Number of vehicles K")
        axi.set_ylabel(f"{MEASURES[col][0]} ({MEASURES[col][1]})", fontsize=8)
        axi.grid(True, color="0.9", lw=0.6)
    ax[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "stage7_measurements.png", dpi=140)
    plt.close(fig)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
