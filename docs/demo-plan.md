# Live demo plan (Windows PC)

For when the professor says "show me your code running". Total live time: about **5 minutes**.
All times below were measured on a 4-core Linux machine with Python 3.11 and the pinned
`requirements.txt` (2 Nov 2026). A normal laptop should be similar; allow up to 2x.

---

## A. The day before (not live): set up and check once

Open **PowerShell** in `D:\cloud\nf-iscsc-reproduction` (or wherever the repo is).

```powershell
git pull                                   # gets experiments/demo_quick.py and this file
py -3.11 -m venv .venv                     # about 5 s
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt  # about 4 min (downloads CVXPY etc.)
python -m pytest -q                        # full test suite: 189 passed, about 12-13 min
```

- If `Activate.ps1` is blocked ("running scripts is disabled"), run
  `Set-ExecutionPolicy -Scope Process Bypass` once in that window, then activate again.
  Or skip activation and type `.venv\Scripts\python.exe` instead of `python` everywhere.
- If `py -3.11` is not found, use whatever `python --version` gives (3.11 or 3.13 both work).
- Do **not** run the full pytest live: 13 of its 12-13 minutes are in `tests/test_beamforming.py`
  (it checks our solver against CVXPY). Run it the day before and take a screenshot of
  "189 passed".
- Do one full rehearsal of section B and note your own times.

## B. Live, in this order

Activate the venv first (`.venv\Scripts\Activate.ps1`). Run everything from the repo root.

| # | Command | Time | Point at | What to say |
|---|---|---|---|---|
| 1 | `python -m pytest -q --ignore=tests/test_beamforming.py` | 24 s | "174 passed" | "Every model has a test against an analytic value, a limiting case or a symmetry. The 15 beamforming tests take 13 minutes, so I ran the full 189 yesterday (screenshot)." |
| 2 | `python -m pytest -q tests/test_beamforming.py -k "lemma1 or subspace or rate_matches or project or infeasible_when"` | 1 s | "7 passed" | "These are the fast beamforming checks, including the one showing Lemma 1 as printed has the wrong sign." |
| 3 | `python experiments/stage1_nearfield_vs_farfield.py` then `start results\stage1_corr_vs_distance.png` | 10 s | Rayleigh distance 288.1 m; divergence 74 / 55 / 18.5 m | "With 310 antennas at 50 GHz the near field reaches 288 m, close to the paper's 300 m. The plane-wave model drops below 0.9 correlation at 74 m broadside, so on a 100 m road the far-field model is wrong." |
| 4 | `python experiments/stage2_scenario.py` then `start results\stage2_layout_example.png` | 1 s | Min gap per K, all >= 3.5 m | "Cars are Poisson placed on a 100 m x 10 m road with the 3.5 m spacing rule. A plain uniform draw at K = 20 meets the rule with chance 1e-13, which is why we place cars slot by slot." |
| 5 | `python experiments/demo_quick.py` | about 2 min | See below | See below |
| 6 | `python experiments/stage5_computing.py` | 1 s | Fig. 8 values (1.09 GHz at 10 Kb, 10.09 GHz at 100 Kb) | "Fig. 8 matches with C = 1000 cycles/bit. Fig. 9 has the same shape but is 5000x higher; that is a units gap in Eq. (25), written down in the assumptions." |

### Step 5: `experiments/demo_quick.py` (the main live demo)

Small settings, clearly marked in the file. Writes only to `results\demo\`, so the committed
results are never touched. Four parts:

1. **Near vs far field** (instant): correlation 0.19 at 5 m, 0.90 at 74 m, 0.99 at 300 m.
2. **Road** (instant): one layout, smallest gap 7.5 m, same seed gives the same layout.
3. **Tracking**, K = 5, 5 seeds, PF500 + EKF + UKF, 100 slots (about 75 s).
   Say: "PF velocity RMSE is 0.85 m/s, near the paper's 0.8. Angle and distance do not match
   the paper, and our EKF is as good as the PF, which the paper does not show. I report that
   as not reproduced." Then point at the last lines: PF500 and EKF differ from the committed
   full run by about 1e-16, so the result is repeatable. **UKF may differ** (0.69 here): its
   covariance gets close to singular, so tiny rounding differences change its path. Say that
   openly; it is an honest finding about the UKF, not a bug in the demo.
   PF2000 is left out only because it alone takes about 3 minutes per seed.
4. **Assignment + beams** (Algorithms 2 and 3), K = 5, 5 seeds, full settings (about 30 s).
   Point at: "HH beats greedy on rate: True" and "Semantic gain, HH over HH non-semantic: 36.0 %
   (paper claims about 20 %)", and the difference from the committed run (about 2e-16).
   Say: "Our gain is 36 %, not 20 %, so the 20 % claim is not reproduced as a number, though
   the direction holds. The 100 Kb case cannot meet Table III's CPU and latency limits at all."

Measured here: whole script 107 s.

## C. If asked for a longer stage

Show the committed plots instead of running them (they take 15 min to 2 h):
`start results\stage3_deg2_relative.png`, `start results\stage5_main_fig5_7.png`,
`start results\stage6_main_gain.png`, `start results\stage7_measurements.png`.
If they insist on a real run, a short trial of any stage is, for example:
`python experiments/stage5_allocation.py --ks 5 --seeds 0 --schemes greedy HH --tag trial`
(about 15-20 s; writes `results\stage5_trial_raw.csv` only). Always use a new `--tag`, so the
`main` results are not overwritten. Do **not** run `stage3_tracking.py` with fewer `--ks` or
`--seeds` live: it has no `--tag` and would overwrite `results\stage3_*_raw.csv`.

## D. If something goes wrong

- `ModuleNotFoundError`: the venv is not active. Activate it, or use `.venv\Scripts\python.exe`.
- Run from the repo root, not from inside `experiments\`.
- A plot does not open: open the PNG from File Explorer in `results\`.
- Stage 1, 2 and 5-computing rewrite their committed CSV and PNG files. The CSVs come out the
  same (seeded); the PNGs may show as changed in `git status` only because of image metadata.
  `git checkout -- results` puts them back.
