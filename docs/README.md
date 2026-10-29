# Guide to the docs folder

These files record how we read the paper, which values we had to choose, and what each stage found. The main [README](../README.md) gives the overview; come here for the reasons behind a choice or a result.

## What each file is

| File | Read it when you want to know |
|---|---|
| [02-task-description.txt](02-task-description.txt) | What the assignment asks for (the brief) |
| [stage0-notes.md](stage0-notes.md) | The system in plain words, which equations belong to which stage, how the paper's optimisation works, and how the paper's figures map to the brief |
| [05-parameters.md](05-parameters.md) | Every number in the paper's Table III and text, with page numbers, and the Rayleigh distance check |
| [06-assumptions.md](06-assumptions.md) | The 36 gaps: things the paper does not say, the value we chose and why. Code comments refer to these as "gap N" |
| [your-call-decisions.md](your-call-decisions.md) | The six gaps that change results most, and what the project owner chose |
| [stage3-notes.md](stage3-notes.md) | Tracking: how the noise values can be read, and why the beta noise needed a "relative" variant |
| [stage4-notes.md](stage4-notes.md) | CRB: Eqs. (15)-(19) derived step by step, the two fixes needed, and the posterior CRB for the filters |
| [stage5-power-check.md](stage5-power-check.md) | Why the power budget cannot be met as written, and the readings of F and Eq. (25) |
| [stage5-notes.md](stage5-notes.md) | Assignment, beamforming and computing: each equation, each deviation, results vs Figs. 5-9 |
| [stage6-notes.md](stage6-notes.md) | The 20% claim: both baselines, feasibility at 1/10/100 Kb, and why the gain is iota / rho |
| [paper_values/](paper_values/README.md) | Values read off the paper's figures by eye, used only for side-by-side plots |

Stage 1 and Stage 2 have no separate notes; their findings are in [results-log.md](../results-log.md).

## Glossary

| Term | Meaning |
|---|---|
| RSU | Roadside unit: the tower with the antenna array |
| ULA | Uniform linear array: antennas in a straight line, equally spaced (here half a wavelength, 3 mm) |
| Nt, Nr | Antennas at the RSU (310) and at each car (3) |
| Near field / far field | Close to a large array, waves are curved (near field); far away they are nearly flat (far field). The boundary is the Rayleigh distance, about 288 m here |
| Steering vector | The phase pattern across the antennas for a given direction (and, in the near field, distance) |
| ISAC | Integrated sensing and communication: one signal both senses targets and carries data |
| ISCSC | Integrated sensing, computing and semantic communication: ISAC plus computing and semantic coding |
| Semantic communication | Sending the meaning of a message instead of every bit. Rate is scaled by iota / rho |
| rho | Semantic extraction ratio: how much of the message is kept (0.81 to 1). Smaller rho means fewer bits but more compute power |
| iota | Word-to-bit factor (1.1, from Table III) |
| Digital twin (DT) | A live software copy of the road, kept up to date from the tracking results |
| CRB, RCRB | Cramer-Rao bound: the lowest possible variance of an unbiased estimate. RCRB is its square root, in degrees or metres |
| FIM | Fisher information matrix; its inverse gives the CRB |
| PF, EKF, UKF | Particle filter, extended Kalman filter, unscented Kalman filter: three ways to track a moving car from noisy measurements |
| HH | Hybrid heuristic: the paper's assignment method (greedy start + simulated annealing, Algorithm 2) |
| HH non-semantic | HH with semantics switched off (rho = 1, iota = 1, no extraction power): our "conventional ISAC" baseline 1 |
| [18] benchmark | The near-field ISAC beamformer of Wang, Mu and Liu (2023), reference [18] of the paper: baseline 2 |
| SDR | Semidefinite relaxation: replace a beam w by the matrix W = w w^H and drop the rank-one rule, so the problem becomes convex |
| AO | Alternating optimisation: fix some variables, solve for the rest, swap, repeat |
| SA | Simulated annealing: a random search that sometimes accepts a worse answer early on, to avoid getting stuck |
| K | Number of cars on the road (5, 10, 15 or 20) |
| Gap N | Item N in [06-assumptions.md](06-assumptions.md): something the paper leaves open |
