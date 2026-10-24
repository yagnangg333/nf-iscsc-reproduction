"""Stage 5: computing performance, Figs. 8 and 9 (Sec. VI-C).

Fig. 8: minimum CPU frequency vs maximum delay, f = (C D + L) / T (Eqs. 22-24, linear f1, f2).
Fig. 9: DT power vs CPU frequency for D = 1, 10, 100 Kb, under each reading of Eq. (25).

The paper does not give C, L or the frequency grid for these figures. Fig. 8's solid curves
(10 GHz at 0.01 s for 100 Kb, 1 GHz for 10 Kb) fit C = 1000 cycles/bit with L close to 0, so
we use C = 1000 (the low end of Table III). L uses the RCRB levels the paper reports for HH
at K = 5 (1e-3 deg, 5e-3 m) plus E|nu3|. The "exponential" curves are skipped (gap 26).
Paper values are read by eye (docs/paper_values/fig9_by_eye.csv).

Run from the repository root:  python experiments/stage5_computing.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import allocation  # noqa: E402
import config  # noqa: E402

RESULTS = ROOT / "results"
COLOURS = ("#2a78d6", "#eb6834", "#eda100")
C_FIG = 1000.0                                   # cycles/bit, fitted to Fig. 8 (see docstring)
SIZES = (1e3, 1e4, 1e5)                          # bits: 1, 10, 100 Kb
NU3_MEAN = np.sqrt(np.pi * config.NU3_VAR) / 2   # E|nu3| for nu3 ~ CN(0, 1000), about 28 cycles


def fig8():
    t = np.linspace(0.01, 0.1, 10)
    l_paper = allocation.workload(1e-3, 5e-3, NU3_MEAN)
    rows = []
    for dbits in SIZES:
        for tt in t:
            rows.append(dict(data_bits=dbits, t_max_s=tt, f_ghz=allocation.min_frequency(C_FIG, dbits, l_paper, tt) / 1e9,
                             f_ghz_no_sensing=allocation.min_frequency(C_FIG, dbits, NU3_MEAN, tt) / 1e9))
    return pd.DataFrame(rows), l_paper


def fig9():
    f = np.linspace(1e9, 6e9, 6)
    rows = []
    for dbits in SIZES:
        for ff in f:
            row = dict(data_bits=dbits, f_ghz=ff / 1e9)
            for reading in ("energy_per_slot", "as_written", "kappa_f3"):
                row[f"p_mw_{reading}"] = 1e3 * allocation.dt_power(ff, C_FIG, dbits, reading)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    f8, l_paper = fig8()
    f9 = fig9()
    f8.to_csv(RESULTS / "stage5_fig8_cpu_vs_delay.csv", index=False)
    f9.to_csv(RESULTS / "stage5_fig9_power_vs_freq.csv", index=False)
    paper9 = pd.read_csv(ROOT / "docs" / "paper_values" / "fig9_by_eye.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for dbits, col in zip(SIZES, COLOURS):
        g = f8[f8.data_bits == dbits]
        ax[0].plot(g.t_max_s, g.f_ghz, "o-", color=col, ms=4, label=f"{dbits / 1e3:.0f} Kb, ours (linear)")
    ax[0].axhline(config.F_MAX / 1e9, color="0.4", ls="--", lw=1)
    ax[0].text(0.098, config.F_MAX / 1e9 + 0.3, "F^max = 5.8 GHz", ha="right", fontsize=8, color="0.3")
    ax[0].set_xlabel("Maximum time delay (s)")
    ax[0].set_ylabel("Minimum required CPU frequency (GHz)")
    ax[0].set_title(f"Fig. 8 setting: C = 1000 cycles/bit, L = {l_paper / 1e6:.2f}e6 cycles", fontsize=9)
    for dbits, col in zip(SIZES, COLOURS):
        g = f9[f9.data_bits == dbits]
        ax[1].semilogy(g.f_ghz, g.p_mw_energy_per_slot, "o-", color=col, ms=4, label=f"{dbits / 1e3:.0f} Kb, ours (energy/slot)")
        p = paper9[paper9.data_bits == dbits]
        ax[1].semilogy(p.f_ghz, p.p_mw, "x:", color=col, ms=5, label=f"{dbits / 1e3:.0f} Kb, paper (by eye)")
    g = f9[f9.data_bits == SIZES[0]]
    ax[1].semilogy(g.f_ghz, g.p_mw_as_written, "k--", lw=1, label="Eq. (25) as written (any D)")
    ax[1].set_xlabel("CPU frequency (GHz)")
    ax[1].set_ylabel("DT power (mW)")
    ax[1].set_title("Fig. 9 setting: C = 1000 cycles/bit", fontsize=9)
    for a in ax:
        a.grid(True, color="0.9", lw=0.6)
        a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "stage5_fig8_fig9.png", dpi=140)
    plt.close(fig)

    print("Fig. 8 at T = 0.01 s (GHz):", f8[f8.t_max_s == 0.01][["data_bits", "f_ghz"]].to_string(index=False))
    m = f9.merge(paper9, on=["data_bits", "f_ghz"])
    m["ours_over_paper"] = m.p_mw_energy_per_slot / m.p_mw
    print("Fig. 9 ours / paper (energy per slot reading):")
    print(m[["data_bits", "f_ghz", "p_mw", "p_mw_energy_per_slot", "ours_over_paper"]].to_string(index=False))


if __name__ == "__main__":
    main()
