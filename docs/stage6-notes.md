# Stage 6 notes: the 20% semantic rate gain at 1, 10 and 100 Kb

The abstract claims "a 20% improvement" in semantic rate over conventional ISAC at equal sensing accuracy. The paper shows it only in Fig. 5 (HH vs HH non-semantic, "drops by 18%"), never at the 1, 10 and 100 Kb sizes of Figs. 8-9. Running Fig. 5 at those sizes is our extension (docs/stage0-notes.md).

Setup: as Stage 5 (Alg. 2 + Alg. 3, L = |nu3|, F^max = 5.8 GHz, power reading "F in mW, (25) as energy per slot"), but every vehicle's task has the same size D. C and nu3 are drawn as in Stage 5 for the same (K, seed). Schemes: HH, HH non-semantic (rho = 1, iota = 1, no P_Comp), HH with Nr = 1. The [18] benchmark ("Benchmark with Nr = 1") is still to come: it needs the [18] paper. Script: experiments/stage6_semantic_gain.py; plot: experiments/stage6_plots.py.

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
