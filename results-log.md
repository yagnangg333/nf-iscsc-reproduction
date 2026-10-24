# Results log

## Stage 0: paper reading (23 Oct 2026)

- No code. Outputs: 05-parameters.md, 06-assumptions.md, stage0-notes.md.
- Rayleigh distance with Nt = 310, fc = 50 GHz, half-wavelength ULA: Nt^2 lambda/2 = 288 m (2D^2/lambda with D = (Nt-1) lambda/2 gives 286 m). "About 300 m" holds to within 4%.
- 36 gaps listed. The largest: RSU positions, units of Q1/Q2, role of rho = 0.81, and power numbers. With Table III values, Eq. (20) (F = 10) and Eq. (25) each exceed Pt = 25 dBm on their own, so constraint (34e) cannot be met as written.
- Eq. (4)/(13) matrix sizes only work if Gamma is applied element-wise.
- Beamforming: alternating optimisation with SDR, Lemma 1 (log-det), Lemma 2 (low SNR), Schur complement for CRB, bisection for rho, Gaussian randomisation. Assignment: greedy + simulated annealing + tabu list.
- The 20% claim comes from Fig. 5 (HH vs HH non-semantic, "drops by 18%"), not from a 1/10/100 Kb experiment. Those data sizes appear only in Figs. 8-9 (CPU frequency, power).
- Missing files: 02-task-description.txt, 03-comparison-framework.pdf. Useful extra: reference [18] for the ISAC baseline.
- Update (brief received): item 6 mapped to the brief's objectives. With Table III values, 100 Kb tasks break the latency/CPU limits (6.7 GHz needed per vehicle vs 5.8 GHz per RSU), and 10 Kb fails above 4-8 vehicles per RSU. "Conventional ISAC" can mean HH non-semantic (about 22% gap in Fig. 5) or the [18] Nr = 1 benchmark (14% to over 200%).
- Decision (23 Oct): Stage 6 tests the 20% claim against both baselines. The main test is HH with semantics off; the second is the [18] ISAC beamformer with Nr = 1, which needs the [18] PDF.
- 23 Oct: the user does not have 03-comparison-framework.pdf. For Stage 7 we will define our own measurement list, marked as ours rather than the framework's, and the user will confirm it before Stage 7.

## Stage 1: near-field channel (23 Oct 2026)

- Files: config.py, channel.py, tests/test_channel.py, experiments/stage1_nearfield_vs_farfield.py.
- Implements Eq. (5) (Fresnel near-field steering and A^H), Eq. (6) (path loss) and Eq. (4) (received signal, with Gamma applied element-wise, gap 9). Exact spherical and plane-wave vectors added for comparison (ours).
- Tests: 21 passed.
- Divergence (exact vs plane-wave correlation < 0.9): 74 m at broadside, 55 m at 30 deg off broadside, 18.5 m at 60 deg off broadside. Rayleigh distance is 288 m, so the 0.9 point sits near R/4 at broadside.
- Random angles 20-160 deg from the axis, 5 seeds: mean divergence distance 48.3 m (std 1.3 m).
- Eq. (5)'s Fresnel form matches the exact phase to correlation > 0.997 beyond 10 m at every angle, so it is safe for the road (nearest lane 7.5 m from the RSU).
- On the 100 m road, a far-field beam loses more than 0.9 dB at 14.1 % of points (std 1.7 %) with the RSU at the road end (assumed), and at 33.8 % (std 2.9 %) with the RSU at mid-road. The near-field effect on this road depends strongly on where the RSU stands, which the paper does not say.
- Paper has no near/far comparison figure, so there is nothing to match; these numbers are our own.
- Decisions (23 Oct): Stage 1 accepted. All six "your call" recommendations accepted (gaps 5, 10, 14, 21, 26, 31). Code pushed to github.com/yagnangg333/nf-iscsc-reproduction (private).

## Stage 2: road scenario (23 Oct 2026)

- Files: scenario.py, tracking/models.py (Eq. 10), tests/test_scenario.py, experiments/stage2_scenario.py, config.py (heading, centre spacing).
- Tests: 130 passed (21 channel + 109 scenario, most of them spacing/bounds checks over K and seeds).
- Placement: fixed K, uniform positions, 3.5 m bumper gap enforced exactly with a 1-D hard-core "shift" sampler. It has the same law as rejection sampling (checked against brute-force rejection at K = 6), but plain rejection is hopeless here: a uniform layout meets the rule with probability 0.41 at K = 5, 0.01 at K = 10, 3e-6 at K = 15 and 1.5e-13 at K = 20.
- So at K = 15-20 the road is close to full (capacity 24): mean smallest gap 3.72 m (std 0.14) at K = 15 and 3.76 m (std 0.03) at K = 20. Without the rule cars would overlap (negative gaps). "Poisson" here is far more regular than a true Poisson layout.
- Vehicles drive towards -x at 20 m/s (ASSUMED). This is the direction Eq. (10) describes: d shrinks by v dT cos(theta).
- Eq. (10) is a first-order step. Run without noise for 2 s it drifts from the exact straight-line track by up to 0.48 m in distance and 2.2 deg in angle (RSU 1, worst car; mean 0.41 m and 1.8 deg). Fig. 3(b) reports PF range RMSE near 0.1 m, so for Stage 3 the true tracks should come from Eq. (10) itself (the paper's state model), not exact geometry, or the filters will see a model error larger than the paper's results.
- Eq. (10)'s beta update keeps beta * d constant, i.e. beta ~ 1/d, not 1/d^2 as assumed for beta_0 (gap 17). Also sigma_beta^2 = 0.1 (Q1) is far larger than beta itself (~1e-5). To settle in Stage 3.
- With both RSUs at the road ends, RSU 2 sees most cars near endfire (angles -145 to -176 deg), where the near-field effect is weakest.

## Stage 3: tracking filters (23 Oct 2026)

- Files: tracking/models.py (Eq. 9-12), tracking/pf.py (Alg. 1, Eq. 29-33), tracking/ekf.py, tracking/ukf.py, tests/test_tracking.py, experiments/stage3_tracking.py, experiments/stage3_plots.py, docs/stage3-notes.md, docs/paper_values/fig3_by_eye.csv, config.py (noise switches), scenario.py (beta_0 = 1/(4 pi d)).
- Tests: 145 passed (15 new: noise units, noise statistics, seeds, Jacobians vs finite differences, echo and Gram vs channel.py, likelihood vs direct formula, filters with no noise).
- beta: Eq. (10) makes beta ~ 1/d (checked), but the paper calls it a round-trip loss (~1/d^2). Table III puts noise of variance 0.1 on beta itself; with beta ~ 1e-3 that is 100-300x beta per slot. As written, |beta| grows ~2000x in 2 s, the PF collapses (ESS ~1 of 2000), angle RMSE 6-30 deg. So we also ran a "relative" beta noise (ours, not the paper's).
- With relative beta noise and deg^2 angles (5 seeds, K = 5-20): PF2000 angle 0.93-1.11 deg (paper ~1.6-2.1), range 0.23 m (paper ~0.08), velocity 0.80 m/s (paper ~0.8). EKF and UKF are about as good as the PF and nearly flat in K; the paper shows them much worse and degrading with K. Not reproduced.
- rad^2 angles give ~60 deg angle RMSE for every filter, far from Fig. 3(a), which supports the deg^2 reading.
- Times: ordering matches (EKF fastest, PF2000 slowest), absolute values are 100-1000x the paper's (PF500 194 ms at K = 20 vs "below 2 ms"), because Eq. (9) makes each measurement a 310-element vector.
- Eq. (10) + Table III noise drifts from straight-line tracks by RMS ~5 deg, 5.5 m, 6.2 m/s over 2 s; 21 of 500 tracks reach a negative speed.
- Decision (23 Oct): later stages use the "relative" beta noise (our variant, 32% of beta per slot). The as-written run stays in results/ for the report.

## Stage 4: Cramer-Rao bound (24 Oct 2026)

- Files: crb.py, tests/test_crb.py, experiments/stage4_crb.py, docs/stage4-notes.md (Eqs. 15-19 quoted and derived step by step), README.md.
- Paper CRB: two fixes needed to make it a valid FIM. Re{} added to Eq. (16), beta* moved inside Re{} in Eq. (17); B uses d where the paper prints s. Checked with a Monte Carlo ML estimator: MSE/CRB 0.99 +/- 0.12 (angle alone), 1.10 and 1.06 (joint angle and distance).
- Road, equal-power beams, no optimisation: angle RCRB 0.0018-0.0023 deg (Fig. 6 HH: 1e-3 to 1e-2 deg, same range). Distance RCRB 3-4 m median (Fig. 7: 5e-3 to 0.5 m). Distance not reproduced yet.
- Filters vs bound: a one-snapshot CRB is the wrong check (filters use 100 slots and the motion model), so we added a posterior CRB (Tichavsky 1998). Angle RMSE is 1.8-3.6x the bound. Distance and velocity sit on the bound (pooled ratio 0.996-2.0); 26 of 160 single runs dip up to 4% under it, which matches the sampling spread of the optimal Kalman filter on 10-40 tracks (simulated: 0.947-1.052).
- Echo noise: the CRB uses -30 dBm, the tracking model uses 0.04 (Table III). They differ by 46 dB; recorded, not changed.
- Tests: 154 passed (9 new in tests/test_crb.py). Python 3.11 venv rebuilt from requirements.txt; same numbers as before.

## Stage 5: assignment, beamforming and computing (24 Oct 2026)

- Files: allocation.py, beamforming.py, experiments/stage5_allocation.py, stage5_computing.py, stage5_plots.py, tests/test_allocation.py, tests/test_beamforming.py, docs/stage5-notes.md, docs/paper_values/fig5_7_by_eye.csv and fig9_by_eye.csv, config.py.
- Tests: 179 passed (25 new). Lemma 1 as printed has the wrong sign (test shows it); we use -ln|E| directly.
- Latency (34g)/(34h) with L of Eq. (24) as written fails for all 80 final assignments (distance RCRB ~1 m x nu2 = 1.2e8 cycles > F^max T^max). Main runs use L = |nu3|. Not tuned.
- Ordering reproduced: HH beats greedy on rate and both RCRBs at every K; gap 4% at K = 5, 30% at K = 20. Greedy with flip is worst here (better than greedy in the paper).
- Rate 18.0 -> 12.7 bps/Hz (HH, K = 5 -> 20) vs paper 3.8 -> 2.8. Not reproduced (our SNR ~50 dB).
- Angle RCRB 7e-4 to 1.4e-3 deg for HH (paper 1e-3 to 1.3e-2). Distance RCRB ~0.6-1 m mean, 0.2-0.4 m median, flat in K (paper 4e-3 to 2.4 m). Not reproduced.
- Non-semantic drop 26.5% vs paper 18%. It is exactly 1 - rho_LB / iota because rho always hits 0.81.
- Fig. 8 reproduced with C = 1000 (10.1 GHz at 0.01 s for 100 Kb). Fig. 9 shape reproduced (f^2, prop. to D), level 5000x the paper's; the paper's level needs C = 0.2 cycles/bit.
- Paper's own text and figures disagree twice (Fig. 6 greedy at K = 20, Fig. 7 HH at K = 20).

## Stage 6: the 20% semantic gain at 1, 10 and 100 Kb (24 Oct 2026; [18] benchmark still to come)

- Files: experiments/stage6_semantic_gain.py, experiments/stage6_plots.py, docs/stage6-notes.md, docs/paper_values/fig5_nr1_by_eye.csv, beamforming.py (rho warm-start fallback), tests (2 new).
- Feasibility with Table III: 1 Kb works at all K; 10 Kb only at K = 5 (one car needs 0.67-1.33 GHz of 5.8); 100 Kb never (one car needs 6.7 GHz).
- At 1 Kb HH beats HH non-semantic by 36% (35.8-37.2%, std under 2.5%) at equal RCRB (ratios 1.01-1.07). Paper: 21-22% from Fig. 5, "20%" in the abstract. Not reproduced.
- The 36% is exactly iota / rho_LB - 1 = 1.1 / 0.81 - 1, because rho always hits 0.81. The paper's figure would need rho about 0.90.
- HH over HH with Nr = 1: 25% (K = 5) to 50% (K = 20); paper flat 80%. Nr = 1 also has 2.5-3.4x worse RCRB.
- 10 Kb, K = 5: 12 +/- 50%. One seed: SA got stuck on a poor island of feasible assignments (exhaustive check over 32 shows the same best assignment for both schemes).
- Fixed: a warm-started rho could make a feasible SA candidate look infeasible under high P_DT. No effect on Stage 5 or these numbers.
