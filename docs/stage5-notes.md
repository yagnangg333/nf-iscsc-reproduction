# Stage 5 notes: assignment, beamforming, semantic ratio and computing

Files: allocation.py (Eqs. 20, 22-25, Alg. 2), beamforming.py (problem (41), Alg. 3), experiments/stage5_allocation.py, experiments/stage5_computing.py, experiments/stage5_plots.py, tests/test_allocation.py, tests/test_beamforming.py. Power reading as chosen on 24 Oct: F in mW, Eq. (25) read as energy per slot, P_DT = kappa f^2 C D / dT (config.DT_POWER; the other readings stay as switches).

## What we implemented, equation by equation

- Objective (34)/(41a): min sum eps(-eta_S) + (1 - eps) eta, eps = 0.5, eta = CRB(theta) + CRB(d) per served pair (Eqs. 15-19, Stage 4 fixes).
- Semantic rate: Eq. (13), iota / rho times log-det, reported in bps/Hz (log2). Inside the optimiser we use the Lemma 2 linear interference bound, Eq. (39). It is never above the true rate (test).
- Lemma 1, Eq. (36): as printed, min_A Tr(AE) - ln|A| - N equals ln|E|, not ln|E^-1| (test). We minimise -ln|E| directly, which is what Lemma 1 is meant to give, so the auxiliary A of Alg. 3 is not needed.
- CRB constraint: the LMI (40) is equivalent to using the inverse FIM directly; we use the FIM in closed form.
- Power (34e): P_C&S (Eq. 21) + P_Comp (Eq. 20) + P_DT (Eq. 25) <= Pt per RSU.
- Computing (34g)/(34h): f = (C D + L) / T^max per task (the smallest f that meets the delay), sum of f <= F^max per RSU.
- Alg. 2: greedy start (nearest RSU), then SA with T0 = 100, alpha = 0.95, Tmin = 1e-3, n_max = 200, one-vehicle flips, B = -(41a).
- Alg. 3: alternate beams (W, R) and rho, stop when the objective changes by less than 0.1 (EPS_TOL) or after 20 rounds (ASSUMED). Rank one by Gaussian randomisation, 100 draws.

## Deviations (each is ours, not the paper's)

1. Interference from RSU m' reaches vehicle k over that RSU's own channel H_{m',k}. The paper writes H_{m,k} for every term, which is not physical with two RSUs.
2. CRB terms and DT tasks count only for served pairs (xi = 1). Otherwise every vehicle adds a CRB term at both RSUs.
3. One sensing matrix R_m per RSU (the paper indexes it by k and i as well; a sum of PSD matrices is one PSD matrix).
4. Exact subspace reduction: the optimum lies in the span of A, dA/dtheta, dA/dd and H for all vehicles of that RSU. Dimension about 20-25 at K = 5 and about 50 at K = 20 (of Nt = 310). Tests show the FIM, the channel gain and the rate are unchanged.
5. Solver. CVXPY (CLARABEL, SCS) fails at the real SNR: SCS returned beams 31% over the power limit. We use L-BFGS on a factorisation X = (P/S) Y Y^H (Burer-Monteiro), which keeps power and PSD exactly. Checked against CLARABEL on a low-SNR case (test, both SPG and L-BFGS match or beat it).
6. L in Eq. (24) needs the RCRB of the "previous slot". We use the previous AO round (cars move 0.4 m per slot), starting from L = |nu3|.
7. rho: one value per RSU found by bisection on the objective, beams rescaled to the power left after P_Comp. The paper gives no rho update rule.
8. Inside SA each candidate gets a cheaper solve (2 AO rounds, 60 L-BFGS steps, warm start, no randomisation). The final assignment gets the full Alg. 3.
9. Latency (main finding, below): the main runs use the "relaxed" latency, L = |nu3| only. For every final assignment we also check (34g)/(34h) with L as written.
10. Single-threaded BLAS (OMP_NUM_THREADS = 1). Results do not depend on it; it is about 15x faster for these small matrices.

## Main finding: the latency constraint as written is never met

With Table III's nu1 = 243.2e6 cycles/deg and nu2 = 121.6e6 cycles/m, a distance RCRB of 1 m adds 1.2e8 cycles. At T^max = 15 ms that is 8 GHz for one vehicle, more than F^max = 5.8 GHz. Our far, near-endfire cars have distance RCRB of 1-4 m, so every assignment of every scheme at every K fails (34h) (written_latency_ok = 0 in all 80 runs). The paper's own Fig. 7 value at K = 20 (about 2.4 m, or "below 0.5 m" in the text) would also fail. So with L as written problem (34) has no feasible point here, and Figs. 5-7 cannot be produced. All results below use L = |nu3|.

## Results (5 seeds, mean +/- std, results/stage5_main_summary.csv, results/stage5_main_fig5_7.png)

| K | scheme | rate (bps/Hz) | paper | angle RCRB (deg) | paper | distance RCRB (m) | paper |
|---|---|---|---|---|---|---|---|
| 5 | HH | 18.0 +/- 1.1 | 3.75 | 7e-4 | 1.0e-3 | 0.62 | 3.7e-3 |
| 5 | greedy | 17.3 +/- 0.8 | 3.35 | 1.0e-3 | 1.7e-3 | 0.83 | 1.2e-2 |
| 20 | HH | 12.7 +/- 1.0 | 2.77 | 1.4e-3 | 1.25e-2 | 0.97 | 2.4 |
| 20 | greedy | 9.8 +/- 1.5 | 0.88 | 3.2e-3 | 0.7 | 1.55 | 8.0 |
| 20 | greedy flip | 3.8 +/- 1.2 | 1.6 | 5.4e-3 | 9e-2 | 7.3 | 6.3 |

- Ordering: reproduced. HH beats greedy at every K (rate and both RCRBs), and the gap grows with K (4% at K = 5, 30% at K = 20). Greedy with random flip is the worst of our schemes; in the paper it beats greedy. Our flip moves 20% of cars to the far RSU (gap 30), which costs more than greedy's mistakes.
- Rate level: not reproduced. Ours is 4-5 times the paper's. Our SNR is about 50 dB with Table III's Pt, Nt = 310 and -30 dBm noise; the paper does not give enough to explain 3-4 bps/Hz.
- Angle RCRB: same order as the paper for HH (1e-3 deg). Greedy does not blow up with K as in Fig. 6.
- Distance RCRB: not reproduced. Ours is about 1 m and nearly flat in K; the paper goes from 4e-3 m to 2.4 m. Our median is 0.2-0.4 m: a few far endfire cars dominate the mean.
- Non-semantic HH: rate drop 26.5% at every K (paper: 18%). It equals 1 - rho / iota = 1 - 0.81 / 1.1 = 26.4%, because rho settles at its lower bound rho_LB = 0.81. The paper's 18% would mean rho about 0.9. The RCRBs are the same with and without semantics, as in the paper.
- For HH, rho always reaches rho_LB (greedy and greedy with flip end at 0.81-1.0, mean 0.85-1.0; Stage 7 tables). P_Comp is 2.1 mW per vehicle and P_DT is 2-21 mW in total, so beams keep 98% (K = 5) to 90% (K = 20) of Pt. CPU use is 0.6-2.3 GHz, under F^max.

## Figs. 8 and 9 (results/stage5_fig8_fig9.png)

- Fig. 8, linear model: reproduced with C = 1000 cycles/bit (low end of Table III; the paper does not say which C). At 0.01 s we need 10.1, 1.09 and 0.19 GHz for 100, 10 and 1 Kb; the paper shows about 10, 1 and 0.1 GHz. L from the paper's HH RCRBs at K = 5 adds 0.09 GHz. The exponential curves are skipped (gap 26: model not given).
- Fig. 9: shape reproduced (power grows as f^2 and in proportion to D). Level not reproduced: ours is exactly 5000 times the paper's at every point with C = 1000. The paper's numbers fit kappa f^2 D times 10 per second, i.e. C / dT = 10, so C = 0.2 cycles/bit with dT = 20 ms. We did not change C to match.
- Eq. (25) as written (kappa f^3 C) gives 0.1-21.6 W, independent of D, which cannot be Fig. 9.

## Paper inconsistencies found

- Fig. 6 shows greedy angle RCRB about 0.7 deg at K = 20; the text says "approximately 10^1 degrees".
- Fig. 7 shows HH distance RCRB about 2.4 m at K = 20; the text says "below 0.5 m".
- Lemma 1 sign (above). Eq. (25) has no data size, but Fig. 9 depends on it.
