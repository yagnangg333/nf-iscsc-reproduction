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
