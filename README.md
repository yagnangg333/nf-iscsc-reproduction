# Near-field ISCSC reproduction

A Python reproduction of:

> Y. Yang, Y. Ding, J. Wang, Z. Yang, C. Zhu, Z. Zhang, D. Niyato and M. Shikh-Bahaei, "Near-Field Integrated Sensing, Computing and Semantic Communication in Digital Twin-Assisted Vehicular Networks", *IEEE Transactions on Vehicular Technology*, vol. 75, no. 9, 2026. DOI [10.1109/TVT.2026.3683229](https://doi.org/10.1109/TVT.2026.3683229).

This is a university assignment. The goal is not to build a product but to check, with our own code, whether the paper's models and claims hold up. Every model is written from the paper's equations, every number has a source, and every result is compared with the paper's figures. Where our numbers do not match, we say so and explain why.

If you are new here, read this page top to bottom (about 10 minutes), then follow [Where to start reading the code](#where-to-start-reading-the-code).

---

## 1. The idea in plain words

Picture a tall tower (a **roadside unit, RSU**) next to a highway. It has a long strip of 310 small antennas. Every 20 milliseconds it does three jobs at once, using **one radio signal and one power budget**:

1. **Watching (sensing).** The signal bounces off the cars and comes back, like a bat's echo. From the echoes the tower works out where each car is (angle and distance) and how fast it moves.
2. **Thinking (computing).** The tower keeps a live copy of the road in software, called a **digital twin**. Updating that copy costs CPU time and power.
3. **Talking (communicating).** The same signal carries data to the cars. To save airtime it sends only the *meaning* of a message, not every bit of it. This is **semantic communication**: a small neural encoder squeezes the message first, which costs some computing power but needs fewer bits on air.

So the tower has to share one budget between "see well", "think fast" and "talk fast". Better sensing means fewer CPU cycles to fix the twin; more squeezing means a faster link but more compute power. The paper's question is: **how should the tower split its power, point its beams and decide which tower serves which car, so that cars get the most data while the tower still sees them accurately enough?**

### Why "near field"?

With 310 antennas at 50 GHz, the array is about 1 m long. For such a large array, waves from a car only 10 to 100 m away arrive as **curved (spherical) wavefronts**, not flat ones. The flat-wave ("far-field") model that most textbooks use breaks down inside roughly 290 m (the Rayleigh distance). The good side: with curved waves, one tower can tell **both the angle and the distance** of a car, and can focus a beam on a spot, not just a direction. The bad side: the maths for beamforming gets harder.

## 2. What the paper proposes

The paper's system (Sec. II of the paper) has two RSUs, one on each side of a 100 m two-lane road, and K = 5 to 20 cars. It solves three linked problems:

| Problem | Paper's method | Where in our code |
|---|---|---|
| Track each car (angle, distance, speed, echo strength) | Particle filter, Algorithm 1; compared with EKF and UKF | `tracking/` |
| Decide which RSU serves which car | Greedy start + simulated annealing, Algorithm 2 | `allocation.py` |
| Pick beams, the semantic squeeze ratio rho and CPU speed | Alternating optimisation with semidefinite relaxation, Algorithm 3 | `beamforming.py`, `allocation.py` |

Sensing accuracy is measured with the **Cramer-Rao bound (CRB)**, the best accuracy any unbiased estimator can reach (`crb.py`).

The headline claim: the scheme gives **about 20% higher data rate than conventional ISAC (integrated sensing and communication) at the same sensing accuracy**.

## 3. What we found, in one table

"Reproduced" means our numbers show it, not that the code runs. Full details per stage are in [`results-log.md`](results-log.md).

| Paper claim or figure | Our result | Status |
|---|---|---|
| Near-field region about 300 m | 288 m (Rayleigh distance, Nt = 310, 50 GHz) | Reproduced |
| Near vs far field (paper has no figure) | Plane-wave model drops below 0.9 correlation at 74 m (broadside), 55 m (30 deg off), 18.5 m (60 deg off) | Our own result |
| Tracking, Fig. 3: PF velocity RMSE about 0.8 m/s | 0.80 m/s | Reproduced |
| Tracking, Fig. 3: PF angle and range RMSE | Angle 0.93-1.11 deg (paper 1.6-2.1), range 0.23 m (paper 0.08) | Not reproduced |
| Tracking, Fig. 3: EKF and UKF much worse than PF | Our EKF and UKF are about as good as the PF | Not reproduced |
| Tracking time ordering (EKF fastest, PF2000 slowest) | Same ordering, but 100-1000x slower in absolute terms | Partly |
| Angle CRB, Fig. 6 | 7e-4 to 1.4e-3 deg for HH (paper 1e-3 to 1.3e-2); only the low end overlaps | Partly |
| Distance CRB, Fig. 7 | About 0.6-1 m, flat in K (paper 4e-3 to 2.4 m) | Not reproduced |
| HH beats greedy on rate and CRB, Figs. 5-7 | Yes, at every K | Reproduced |
| Rate level, Fig. 5 (3.8 to 2.8 bps/Hz) | 18.0 to 12.7 bps/Hz | Not reproduced |
| Fig. 8 (CPU frequency vs delay) | Matches with C = 1000 cycles/bit | Reproduced |
| Fig. 9 (DT power vs CPU frequency) | Same shape, level 5000x higher | Partly |
| **About 20% rate gain over conventional ISAC at equal accuracy** | At 1 Kb: 36% over "HH non-semantic" (paper's Fig. 5: 21-22%), and 28-36% over the [18] ISAC benchmark (paper's Fig. 5: 15% to 219%). At 10 Kb (K = 5 only): 14% over [18] | **Not reproduced** |
| Gain at 1, 10, 100 Kb tasks | 1 Kb works at all K; 10 Kb only at K = 5; 100 Kb never fits Table III's CPU and latency limits | Our own result |

Two root causes explain most of the gaps:

- **The path loss of Eq. (6) has no wavelength in it**, so our link SNR is about 50 dB. That pushes rates far above the paper's.
- **The semantic ratio rho always hits its lower bound 0.81.** The rate gain is then exactly iota / rho - 1 = 1.1 / 0.81 - 1 = 36%, fixed by two table values, not by the optimisation.

Many details are missing from the paper (RSU positions, noise units, power units and more). We listed 36 such gaps and wrote down every choice. See [`docs/06-assumptions.md`](docs/06-assumptions.md).

## 4. Quick start

You need **Python 3.11** (3.13 also works with the pinned versions).

```bash
git clone https://github.com/yagnangg333/nf-iscsc-reproduction.git
cd nf-iscsc-reproduction
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the tests (189 tests, about 10 minutes on one core):

```bash
pytest -q
```

Run a fast experiment to see something work (a few seconds each):

```bash
python experiments/stage1_nearfield_vs_farfield.py   # near vs far field
python experiments/stage2_scenario.py                # road layout
python experiments/stage5_computing.py               # Figs. 8 and 9
```

Plots and CSV files appear in `results/`. All the results from our full runs are already committed there, so you can look at them without running anything.

## 5. How the code is laid out

```
config.py            every parameter, each tagged with its source (Table III, Eq., Sec. or ASSUMED)
channel.py           Stage 1: near-field steering vector, path loss, channel
scenario.py          Stage 2: road, RSUs, car placement and movement
tracking/
  models.py          Stage 3: car motion model and echo measurement model
  pf.py              Stage 3: particle filter (Algorithm 1)
  ekf.py             Stage 3: extended Kalman filter
  ukf.py             Stage 3: unscented Kalman filter
crb.py               Stage 4: Cramer-Rao bound (paper's and a posterior version for the filters)
allocation.py        Stage 5: computing power, latency, DT power, car-to-RSU assignment (Algorithm 2)
beamforming.py       Stage 5: beams, semantic ratio rho, the inner problem (Algorithm 3)
baselines.py         Stage 6: the conventional ISAC benchmark from reference [18]
experiments/         one script per stage, figure or table
tests/               pytest checks against analytic values, limits and symmetries
results/             CSV (raw numbers) and PNG (plots) from every experiment
docs/                paper reading notes, parameters, assumptions, per-stage notes
results-log.md       what each stage found, in short
```

### Which file implements which paper equation

| Paper | What it is | Code |
|---|---|---|
| Eq. (3) | Transmit covariance Rx = W W^H + R R^H | `crb.py`, `beamforming.py` |
| Eq. (4) | Channel G = Gamma (element-wise) A^H, and the received signal at a car | `channel.channel_matrix`, `channel.received_signal` |
| Eq. (5) | Near-field steering vectors, cross term H and A^H | `channel.nf_steering`, `channel.cross_term`, `channel.steering_matrix_H` |
| Eq. (6) | Per-element path loss Gamma | `channel.path_loss_matrix` |
| Eqs. (9), (11), (12) | Echo measurement model and noise | `tracking/models.py` (`g2`, `measure`, `log_likelihood`) |
| Eqs. (10), (11) | Car motion (state) model | `tracking/models.py` (`g1`, `step`, `propagate`) |
| Eq. (13) | Semantic rate | `beamforming.rates` |
| Eqs. (15)-(19) | CRB of angle and distance | `crb.fim_paper`, `crb.crb_paper` |
| Eq. (20) | Semantic extraction power P_Comp | `allocation.comp_power` |
| Eq. (21) | Beam power P_C&S | `beamforming.beam_power` |
| Eqs. (22)-(24) | Latency and DT workload | `allocation.latency`, `allocation.workload` |
| Eq. (25) | Digital twin CPU power P_DT | `allocation.dt_power` |
| Eqs. (26)-(33), Alg. 1 | Particle filter | `tracking/pf.py` |
| Eq. (34) | The full optimisation problem | `beamforming.objective` + `allocation.py` |
| Eqs. (36)-(41), Alg. 3 | Convex inner problem, AO loop, Gaussian randomisation | `beamforming.inner_problem`, `beamforming.solve_beams`, `beamforming.rho_step`, `beamforming.randomise` |
| Alg. 2 | Greedy + simulated annealing assignment | `allocation.greedy`, `allocation.hybrid_heuristic` |
| [18] problem (16), (20), (21) | Conventional ISAC benchmark | `baselines.py` |

Each module's top docstring quotes the equations it implements and lists any deviation.

### How the pieces fit

```
config.py ──► scenario.py ──► (theta, d) of every car from every RSU
                                   │
              channel.py ◄─────────┘   near-field channel for each car
                 │
     ┌───────────┼────────────────────────┐
     ▼           ▼                        ▼
 tracking/     crb.py                beamforming.py ◄── allocation.py
 (Stage 3)    (Stage 4)              (inner problem)    (assignment, power, CPU)
 follow cars  best possible          beams + rho         which RSU serves which car
 over time    accuracy                      │
                                            ▼
                                     baselines.py (Stage 6) → compare rates at equal CRB
```

## 6. Running every stage

### Why the work is split into stages

The paper describes one big system. We built it in eight stages, and each stage had to pass its tests and be checked by the project owner before the next one started:

| Stage | What it builds | Brief objective |
|---|---|---|
| 0 | Read the paper: parameters, gaps, no code | All |
| 1 | Near-field channel, compared with far field | System model, near vs far field |
| 2 | Road, RSUs and cars | System model |
| 3 | PF, EKF and UKF tracking | Tracking filters |
| 4 | Cramer-Rao bound | CRB as the sensing benchmark |
| 5 | Assignment, beamforming, computing | Resource allocation and beamforming |
| 6 | The 20% claim at 1, 10 and 100 Kb | Headline claim |
| 7 | Measurements and report | Report and measurements |

Why this order:

1. **Each stage needs the one before it.** Tracking needs the channel and the road. The CRB needs the channel and the tracking results. Beamforming needs the CRB. The 20% claim needs everything.
2. **A bug is caught where it starts.** If everything were built at once and the final rate came out wrong, there would be no way to tell whether the channel, the filter or the optimiser was at fault. For example, the wrong sign in Lemma 1 was caught in Stage 5 instead of hiding inside the Stage 6 numbers.
3. **Each objective of the brief can be checked on its own.** The table above shows the mapping.
4. **Every choice has a known place.** Each gap in the paper (noise units, power units and so on) was settled and written down at a fixed stage, in `results-log.md` and `docs/`. That makes the work easy to explain in the same order.

### Commands

All commands run from the repository root. Times are for 4 cores. `OMP_NUM_THREADS=1` keeps each worker on one core (faster here, and makes run times comparable).

| Stage | Command | Time | Main outputs in `results/` |
|---|---|---|---|
| 1 Near vs far field | `python experiments/stage1_nearfield_vs_farfield.py` | seconds | `stage1_*.csv`, `stage1_corr_vs_distance.png` |
| 2 Road scenario | `python experiments/stage2_scenario.py` | seconds | `stage2_*.csv`, `stage2_layout_example.png` |
| 3 Tracking (Fig. 3) | `python experiments/stage3_tracking.py --angle-unit deg2 --beta-noise relative` then `python experiments/stage3_plots.py deg2_relative` | about 15 min | `stage3_deg2_relative_raw.csv`, `.png` |
| 4 CRB (needs Stage 3) | `python experiments/stage4_crb.py` | about 3 min | `stage4_*.csv`, `stage4_*.png` |
| 5 Assignment and beams (Figs. 5-7) | `OMP_NUM_THREADS=1 python experiments/stage5_allocation.py --workers 4 --tag main` then `python experiments/stage5_plots.py --tag main` | about 1 h | `stage5_main_*.csv`, `stage5_main_fig5_7.png` |
| 5 Computing (Figs. 8-9) | `python experiments/stage5_computing.py` | seconds | `stage5_fig8_*.csv`, `stage5_fig9_*.csv`, `stage5_fig8_fig9.png` |
| 6 Semantic gain at 1/10/100 Kb | `OMP_NUM_THREADS=1 python experiments/stage6_semantic_gain.py --workers 4 --tag main` | about 40 min | `stage6_main_raw.csv`, `stage6_main_gain.csv`, `stage6_main_feasibility.csv` |
| 6 [18] benchmark, R_min sweep | `OMP_NUM_THREADS=1 python experiments/stage6_benchmark18.py --workers 4 --tag main` | about 1.5 h | `stage6_main_benchmark18_*.csv` |
| 6 [18] at HH's own rates | `OMP_NUM_THREADS=1 python experiments/stage6_benchmark18.py --workers 4 --tag main --matched` then `python experiments/stage6_plots.py --tag main` | about 30 min | `stage6_main_benchmark18_matched.csv`, `stage6_main_gain.png` |
| 7 Measurements, all six schemes | `OMP_NUM_THREADS=1 python experiments/stage5_allocation.py --workers 4 --tag stage7` then `python experiments/stage7_measurements.py --tag stage7` | about 2 h | `stage7_measurements.csv`, `.md`, `.png` |

Useful options: `--ks 5 10` (fewer car counts), `--seeds 0 1` (fewer seeds), `--workers N`. For a quick trial of a long stage, use for example `--ks 5 --seeds 0`.

Stage 3 also takes `--angle-unit deg2|rad2` and `--beta-noise absolute|relative`, because the paper does not give the units of its noise values. The results for all four readings are in `results/`, and [`docs/stage3-notes.md`](docs/stage3-notes.md) explains them.

## 7. Rules the code follows

- **SI units inside the code** (m, s, Hz, W, rad). Degrees and dB appear only on plots and in tables.
- **Angle theta is measured from the array axis** (the road direction), so 90 deg is broadside (straight out from the tower).
- **Every parameter lives in `config.py`** with its source. Values the paper does not give are marked `ASSUMED` with a reason and a gap number from `docs/06-assumptions.md`.
- **Every random step takes a seed.** Every experiment runs at least 5 seeds (`config.SEEDS`) and saves raw numbers to `results/*.csv`, with mean and std.
- **Every model has a test** against an analytic value, a limiting case or a symmetry (for example: near-field steering tends to far-field as distance grows; our CRB matches a Monte Carlo maximum-likelihood estimator).
- **Deviations from the paper are named.** Where the paper's equation cannot be right as printed (for example the sign in Lemma 1, or a missing Re{} in Eq. (16)), the docstring says what we changed and the stage notes show why.

## 8. Where to start reading the code

A good order for a new developer:

1. `config.py`: all the numbers, so you know the setting.
2. `channel.py` and `tests/test_channel.py`: the near-field model, short and self-contained.
3. `scenario.py`: where cars and RSUs are.
4. `tracking/models.py`, then `tracking/pf.py` (the paper's method), then `ekf.py` and `ukf.py`.
5. `crb.py`: how sensing accuracy is measured.
6. `allocation.py`, then `beamforming.py` (the biggest file; start at `inner_problem` at the bottom and read upwards).
7. `baselines.py`: the comparison scheme.
8. Any script in `experiments/` to see how one figure is produced end to end.

For background, read [`docs/stage0-notes.md`](docs/stage0-notes.md) (the system in plain words and the equations per stage), then the stage notes as you reach each module. [`docs/README.md`](docs/README.md) lists every document and has a glossary of terms.

## 9. Dependencies

NumPy, SciPy, Matplotlib, Pandas and pytest, with exact versions pinned in `requirements.txt` so results repeat. CVXPY (with the free solvers SCS and CLARABEL) is used **only to check** our own beamforming solver on small cases: at the real SNR it was not accurate enough, so the experiments use our own L-BFGS solver (see [`docs/stage5-notes.md`](docs/stage5-notes.md)). Newer CVXPY versions reach a slightly different optimum and can fail two tight comparison tests, which is why the versions are pinned.
