# Near-field ISCSC reproduction

Python reproduction of Y. Yang et al., "Near-Field Integrated Sensing, Computing and Semantic Communication in Digital Twin-Assisted Vehicular Networks", IEEE Transactions on Vehicular Technology, vol. 75, no. 9, 2026, DOI 10.1109/TVT.2026.3683229.

The work is done in stages; `results-log.md` has the notes for each stage. Paper parameters, gaps and assumptions are in `docs/`.

## Install

Python 3.11.

```
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CVXPY is outside the base library list (NumPy, SciPy, Matplotlib, Pandas) and was added with the project owner's approval, with the free solvers SCS and CLARABEL. In Stage 5 it failed at the real SNR, so the beamforming uses our own L-BFGS solver; CVXPY (CLARABEL) is used only to check our solvers on small cases: Stage 5 beamforming and the [18] benchmark in Stage 6 (docs/stage5-notes.md, docs/stage6-notes.md).

## Run

All commands from the repository root.

```
pytest -q                                               # all tests
python experiments/stage1_nearfield_vs_farfield.py      # Stage 1
python experiments/stage2_scenario.py                   # Stage 2
python experiments/stage3_tracking.py --angle-unit deg2 --beta-noise relative   # Stage 3 (about 15 min on 4 cores)
python experiments/stage3_plots.py deg2_relative
python experiments/stage4_crb.py                                               # Stage 4 (about 3 min; needs the Stage 3 CSV)
OMP_NUM_THREADS=1 python experiments/stage5_allocation.py --workers 4 --tag main   # Stage 5, Figs. 5-7 (about 1 h on 4 cores)
python experiments/stage5_plots.py --tag main
python experiments/stage5_computing.py                                         # Stage 5, Figs. 8-9 (seconds)
OMP_NUM_THREADS=1 python experiments/stage6_semantic_gain.py --workers 4 --tag main   # Stage 6 (about 40 min on 4 cores)
OMP_NUM_THREADS=1 python experiments/stage6_benchmark18.py --workers 4 --tag main             # [18] R_min sweep (about 1.5 h)
OMP_NUM_THREADS=1 python experiments/stage6_benchmark18.py --workers 4 --tag main --matched   # [18] at HH's rates (about 30 min)
python experiments/stage6_plots.py --tag main
```

Outputs go to `results/` (CSV and PNG).

## Layout

```
config.py          all parameters, with their source (Table III, equation, section or ASSUMED)
channel.py         near-field (Eq. 5) and far-field steering vectors, path loss (Eq. 6), channel (Eq. 4)
scenario.py        road and vehicles (Stage 2)
tracking/          PF, EKF, UKF (Stage 3)
crb.py             Cramer-Rao bounds: paper Eqs. (15)-(19) and a posterior CRB for the filters (Stage 4)
allocation.py      simulated annealing assignment, compute allocation (Stage 5)
beamforming.py     beamforming and semantic extraction ratio (Stage 5)
baselines.py       conventional ISAC baselines (Stage 6)
experiments/       one script per figure or table
tests/             pytest tests
results/           CSV and PNG outputs
docs/              Stage 0 parameter table, gaps, notes, task brief
```

## Conventions

- SI units inside the code (m, s, Hz, W, rad). Degrees and dB only on plots.
- theta is measured from the array axis (the road direction); 90 deg is broadside.
- Every random process takes a seed; experiments run at least 5 seeds.

Stage 3 options: `--angle-unit deg2|rad2` and `--beta-noise absolute|relative` choose how the Table III noise values are read (see `docs/stage3-notes.md`); `--ks`, `--seeds` and `--workers` set the sweep. Set `OMP_NUM_THREADS=1` when running with several workers so the timing is per core.
