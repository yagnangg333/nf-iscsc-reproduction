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

CVXPY is used for the beamforming optimisation (Stage 5). It is outside the base library list (NumPy, SciPy, Matplotlib, Pandas) and was added with the project owner's approval. It runs with the free pip-installable solvers SCS and CLARABEL; no licence is needed.

## Run

All commands from the repository root.

```
pytest -q                                               # all tests
python experiments/stage1_nearfield_vs_farfield.py      # Stage 1
```

Outputs go to `results/` (CSV and PNG).

## Layout

```
config.py          all parameters, with their source (Table III, equation, section or ASSUMED)
channel.py         near-field (Eq. 5) and far-field steering vectors, path loss (Eq. 6), channel (Eq. 4)
scenario.py        road and vehicles (Stage 2)
tracking/          PF, EKF, UKF (Stage 3)
crb.py             Cramer-Rao bound (Stage 4)
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
