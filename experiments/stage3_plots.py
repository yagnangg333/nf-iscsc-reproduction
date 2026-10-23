"""Stage 3 plots: our Fig. 3 (mean +/- std over seeds) next to the paper's bars read by eye.

Run after stage3_tracking.py:  python experiments/stage3_plots.py deg2_relative
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FILTERS = ("PF500", "PF2000", "EKF", "UKF")
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")      # dataviz categorical slots 1-4
PANELS = (("rmse_theta_deg", "Angle RMSE (deg)"), ("rmse_d_m", "Range RMSE (m)"),
          ("rmse_v_mps", "Velocity RMSE (m/s)"), ("time_per_slot_ms", "Time per slot (ms)"))


def main(tag):
    raw = pd.read_csv(RESULTS / f"stage3_{tag}_raw.csv")
    paper = pd.read_csv(ROOT / "docs" / "paper_values" / "fig3_by_eye.csv")
    ks = sorted(raw.K.unique())
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    width = 0.2
    for col, (metric, label) in enumerate(PANELS):
        for row, (src, title) in enumerate(((raw, "This reproduction"), (paper, "Paper Fig. 3 (read by eye)"))):
            ax = axes[row, col]
            for j, (f, c) in enumerate(zip(FILTERS, COLOURS)):
                g = src[src["filter"] == f].groupby("K")[metric]
                mean = g.mean().reindex(ks)
                std = g.std().reindex(ks) if src is raw else None
                pos = np.arange(len(ks)) + (j - 1.5) * width
                ax.bar(pos, mean.values, width * 0.9, color=c, label=f,
                       yerr=None if std is None else std.values, error_kw=dict(lw=0.8, ecolor="0.3"))
            if metric == "time_per_slot_ms" and src is raw:
                ax.set_yscale("log")
            ax.set_xticks(np.arange(len(ks)), [str(k) for k in ks])
            ax.grid(True, axis="y", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
            if row == 1:
                ax.set_xlabel("Number of vehicles K")
            ax.set_ylabel(label, fontsize=8)
            ax.set_title(title, fontsize=8, color="0.3")
    for col in range(3):                                   # same y-scale in both rows, so bars compare directly
        top = max(axes[0, col].get_ylim()[1], axes[1, col].get_ylim()[1])
        for row in range(2):
            axes[row, col].set_ylim(0, top)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"Tracking vs K: reproduction ({tag}, mean +/- std over seeds) and paper", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / f"stage3_{tag}.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "deg2_relative")
