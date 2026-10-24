# Stage 6 notes: the 20% semantic rate gain at 1, 10 and 100 Kb

The abstract claims "a 20% improvement" in semantic rate over conventional ISAC at equal sensing accuracy. The paper shows it only in Fig. 5 (HH vs HH non-semantic, "drops by 18%"), never at the 1, 10 and 100 Kb sizes of Figs. 8-9. Running Fig. 5 at those sizes is our extension (docs/stage0-notes.md).

Setup: as Stage 5 (Alg. 2 + Alg. 3, L = |nu3|, F^max = 5.8 GHz, power reading "F in mW, (25) as energy per slot"), but every vehicle's task has the same size D. C and nu3 are drawn as in Stage 5 for the same (K, seed). Schemes: HH, HH non-semantic (rho = 1, iota = 1, no P_Comp), HH with Nr = 1. The [18] benchmark ("Benchmark with Nr = 1") is in baselines.py (section below). Scripts: experiments/stage6_semantic_gain.py, experiments/stage6_benchmark18.py; plot: experiments/stage6_plots.py.

## Feasibility (results/stage6_main_feasibility.csv)

| D | K = 5 | K = 10-20 |
|---|---|---|
| 1 Kb | feasible | feasible |
| 10 Kb | feasible | no feasible assignment (CPU) |
| 100 Kb | infeasible: one vehicle alone needs more than F^max | same |

100 Kb: f = C D / T^max = 1000 x 1e5 / 0.015 = 6.7 GHz at the lowest C, above F^max = 5.8 GHz. 10 Kb: one vehicle needs 0.67-1.33 GHz, so an RSU holds about 4-8 vehicles; K = 10 already fails for every seed. These are what the Table III numbers give; the paper reports no rate at these sizes.

## Gain at 1 Kb (results/stage6_main_gain.csv, results/stage6_main_gain.png; 5 seeds, per-seed ratio)

| K | HH over HH non-semantic | paper (Fig. 5) | HH over HH Nr = 1 | paper (Fig. 5) |
|---|---|---|---|---|
| 5 | 35.8 +/- 0.6% | 22% | 25 +/- 14% | 81% |
| 10 | 36.3 +/- 0.3% | 21% | 34 +/- 8% | 80% |
| 15 | 37.2 +/- 2.4% | 22% | 41 +/- 9% | 82% |
| 20 | 36.1 +/- 0.6% | 21% | 50 +/- 20% | 81% |

- Equal sensing accuracy holds against the non-semantic design: RCRB ratios (HH / non-semantic) are 1.01-1.07 for angle and 1.03-1.06 for distance.
- The semantic gain is 36%, not 20%. It is set by iota / rho_LB = 1.1 / 0.81 = 1.358: rho always reaches its lower bound, so the gain is fixed by Table III and does not depend on K, as in the paper (flat 21-22%). The paper's 22% would need rho about 0.90. Not reproduced, and we did not change rho_LB.
- Against our own design with Nr = 1 the gain grows with K (25% to 50%); the paper shows a flat 80%. With Nr = 1 the angle and distance RCRBs are also 2.5-3.4 times worse, so this comparison is not at equal sensing accuracy.
- Gain at 10 Kb, K = 5: 12 +/- 50% over non-semantic. One seed (4) drives the spread. There P_DT takes 320-480 mW of the 632 mW total, 19 of the 32 assignments are infeasible, and the feasible ones form islands that one-vehicle flips cannot cross. Semantic HH started (greedy is infeasible, so from a random restart) on a poor island and stayed there (2.4 bps/Hz), while non-semantic HH reached the best assignment (10.3 bps/Hz). Scoring all 32 assignments shows that the best one is the same for both, and the semantic rate there is 13.8 bps/Hz. This is Alg. 2's behaviour, not a scoring error (the cheap SA score, -2179.7, matches the full solve, -2177.0). We kept Alg. 2 as written.

## Code change found while doing this

Inside SA, rho is warm-started from the previous candidate (0.81). When P_DT is large, P_Comp at 0.81 can exceed what is left of Pt even though rho = 1 (no P_Comp) would fit, and the candidate was wrongly called infeasible. inner_problem now falls back to rho = 1 on that RSU (test: test_warm_rho_never_makes_a_feasible_case_infeasible, which fails on the old code). Stage 5 could not hit this (P_DT + P_Comp under 65 mW there), and a rerun of all 10 Kb cases gave identical numbers.

## The [18] benchmark ("Benchmark with Nr = 1")

[18] is Z. Wang, X. Mu, Y. Liu, "Near-field integrated sensing and communications", IEEE Commun. Lett. 27(8), 2023. It minimises tr(CRB) of one target's distance and angle subject to a minimum rate per single-antenna user (problem 16), and solves the SDR (20) exactly, then rebuilds rank-one beams with Eq. (21). It has no semantics and one BS. baselines.py implements it as written, with these additions to fit the main paper's setting (each is ours):

1. Both RSUs are solved jointly. The interference term in (18) sums over both RSUs, each over its own channel (as in Stage 5). It stays linear, so the problem stays convex.
2. Every served vehicle is a user and a target; tr(CRB) is the sum over targets of CRB(d) + CRB(theta), i.e. the main paper's eta ([18] footnote 1 says the multi-target CRB has the same form).
3. P_max = Pt - P_DT per RSU (no P_Comp, as there is no semantic extraction).
4. R_min,k is not given in the main paper. We sweep it (0.5-8 bit/s/Hz) and also run a matched case (below).
5. Assignment: the final assignment of HH with Nr = 1 for the same (K, seed). The main paper does not say which assignment its benchmark used.

Solver. CVXPY (CLARABEL) solves (20) as written, but only accurately on small cases: tr(CRB) spans about 1e-10 to 1e2 here, and at K = 20 CVXPY runs out of memory while building the problem. So the runs use the Stage 5 factorised L-BFGS solver with eps = 0 (its objective is then exactly tr(CRB)) and the rate constraints by an augmented Lagrangian. The rate constraint is written as SINR / (2^R_min - 1) - 1 >= 0, the same set as (18); an earlier form, gamma s / (t + 1) - 1, lost up to 0.6 bit/s/Hz at high R_min and was replaced. Checks: the two solvers agree within 0.25% where CLARABEL reports "optimal" (tests/test_baselines.py, 7 tests), and [18] given HH's own per-vehicle rates always reaches an eta no larger than HH's, as an optimal solver must.

### Results

- R_min sweep (results/stage6_main_benchmark18_raw.csv): rates meet R_min after Eq. (21) up to 4 bit/s/Hz at every K; 6 and 8 fail for some seeds at K = 15-20 (valid fraction 0.87 and 0.60). tr(CRB) grows only slowly with R_min (mean distance RCRB 1.50 -> 1.58 m at K = 5 from R_min 0.5 to 8): the worst-placed cars dominate it.
- HH with Nr = 3 is more accurate (smaller eta) than [18] at every R_min we tried, at every K. So against the proposed design, the benchmark loses on both rate and sensing; an "equal accuracy" rate gain cannot be defined there.
- Matched comparison (results/stage6_main_benchmark18_matched.csv): [18] is given, as R_min,k, each vehicle's Shannon rate from HH with Nr = 1 on the same assignment. It meets every rate (largest shortfall 1.4e-4 bit/s/Hz) with a smaller eta: ratio 0.97 (K = 5), 0.93 (10), 0.80 (15), 0.75 (20), 0.995 at 10 Kb. So at the same bits per channel use, [18] senses as well or better. HH's whole advantage is the semantic factor iota / rho: 35.8% at K = 5 and 10, 33.6 +/- 4.8% at 15, 28.0 +/- 11.3% at 20 (rho rises above 0.81 for some seeds), 14 +/- 7% at 10 Kb (rho near 1, because P_DT takes most of Pt).
- Paper (Fig. 5, by eye): HH with Nr = 1 over the benchmark is 15% at K = 5, rising to 219% at K = 20. Not reproduced: in our runs the beamforming of [18] is at least as good as ours at equal rate, and the gap does not grow with K.

### What this means for the 20% claim

Against both readings of "conventional ISAC" (HH without semantics, and [18]), the semantic rate gain at equal or better sensing accuracy comes entirely from iota / rho in Eq. (13), not from the beamforming or the assignment. With Table III's iota = 1.1 and rho_LB = 0.81 it is 36%, not 20%; it falls when P_DT leaves too little power for P_Comp (14% at 10 Kb). We did not tune rho_LB or iota.
