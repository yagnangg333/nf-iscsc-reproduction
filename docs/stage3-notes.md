# Stage 3 notes: tracking filters

## What the paper says about beta and its noise

- Sec. II-C, Eq. (7) and (9): beta is the "round-trip path loss" (bold, a matrix, in Eq. 7; a scalar in the state vector).
- Eq. (10), last line: beta_i = beta_{i-1} (1 + v dT cos(theta) / d) + u_beta. To first order this is exactly how 1/d changes as d shrinks by v dT cos(theta), so Eq. (10) treats beta as proportional to 1/d. A physical round-trip loss would fall as 1/d^2 (each way 1/d). The equation and the wording disagree; we follow the equation.
- Eq. (10)-(12): u_beta is noise on beta itself, with variance sigma_beta^2 from Q1. Table III gives Q1 = [0.02, 0.3, 1, 0.1] with no units and cites [33], [35], [54]. The paper does not define u_beta on any other quantity (such as a log or relative scale).
- The paper never gives a value or scale for beta. We set beta_0 = 1/(4 pi d) (from Eq. 6, Gamma = 1/(sqrt(4 pi) d)); that is 9e-4 to 3e-3 on this road.

So, taken literally, u_beta has standard deviation 0.32 per 20 ms slot, about 100 to 300 times beta itself. In our runs |beta| grows by about 2000x in 2 s. The most likely explanation is that the cited works use a normalised beta of order 1, and Table III copies their noise values without rescaling. We cannot confirm this without [33], [35] and [54].

## How noise is applied (all switches in config.py)

- Truth: q_i = g1(q_{i-1}) + u_i (Eq. 11), u ~ N(0, Q1), run separately for every RSU-vehicle pair, 100 slots, own seed per pair.
- Q1 angle entry: "deg2" (accepted reading) = 0.02 deg^2; "rad2" = 0.02 rad^2.
- Q1 beta entry: "absolute" = N(0, 0.1) on beta, as written; "relative" = N(0, 0.1 beta^2), our variant (32% of beta per slot). Not in the paper.
- Measurement (Eq. 9 + Sec. II-C): Nt complex echo values with CN(0, 0.04) noise each; d_hat = d + N(0, 0.06); v_hat = v + N(0, 1). Q2 used as variances.

## Deviations from the paper (all deliberate, all recorded)

1. Truth tracks come from Eq. (10) per RSU-vehicle pair, so RSU 1 and RSU 2 track two different noisy copies of each car.
2. The probe beam for slot i points at g1(true state at i-1), shared by all filters. The paper steers it at each filter's own prediction (Sec. II-A). Ours keeps the measurements identical across filters.
3. Matched filter (Eq. 8) not simulated: Remark 1 lets tau and mu be taken as exact before the d and v noise is added.
4. EKF and UKF are the textbook forms (the paper only cites [2] and [55]). UKF uses alpha = 1e-3, beta = 2, kappa = 0; the Cholesky step fails in some slots, and the filter then coasts on g1 (counted as failed steps).
5. Eq. (30) is evaluated with the normalised weights before resampling (Alg. 1 puts it after).
6. RMSE pools all vehicles, both RSUs and slots 1 to 100. The paper does not define its averaging.
7. Time is the filter time per slot for all K vehicles of one RSU (Python, one core). The paper does not give its hardware or language.

## Results: paper vs reproduction

See results/stage3_*_raw.csv (one row per K, seed, filter) and results/stage3_*.png. Paper values are read by eye (docs/paper_values/fig3_by_eye.csv).

"relative" beta noise (our variant; the only reading where every filter tracks), angle deg^2, mean over 5 seeds:

| Metric | Paper PF2000 | Ours PF2000 | Paper EKF | Ours EKF |
|---|---|---|---|---|
| Angle RMSE, K = 5 / 20 (deg) | 1.6 / 2.1 | 0.93 / 1.03 | 1.6 / 5.8 | 1.09 / 1.28 |
| Range RMSE, K = 5 / 20 (m) | 0.08 / 0.08 | 0.226 / 0.227 | 0.85 / 1.8 | 0.226 / 0.226 |
| Velocity RMSE, K = 5 / 20 (m/s) | 0.80 / 0.82 | 0.81 / 0.80 | 2.1 / 3.6 | 0.79 / 0.79 |
| Time per slot, K = 5 / 20 (ms) | 0.9 / 3.45 | 300 / 1127 | ~0 / ~0 | 2.1 / 7.5 |

As written ("absolute" beta noise): beta grows about 2000x in 2 s, the PF collapses onto one particle (effective sample size about 1.1 to 1.6 of 2000), and errors are 6 to 30 deg in angle and 2 to 6 m in range. Not comparable to Fig. 3.

Angle in rad^2 (sensitivity, K = 5 and 20): the true angle wanders 8 deg per slot, and every filter has about 60 deg angle RMSE. Fig. 3(a) shows 1.6 to 9 deg, so the rad^2 reading does not fit the paper. This supports the deg^2 reading (gap 14) on evidence from the paper, not by tuning.

## Verdict per claim of Sec. VI-A

- PF velocity RMSE about 0.8 m/s: reproduced (0.80-0.81).
- PF angle RMSE of 1.6-2.5 deg: same order, ours lower (0.9-1.4 deg). Partly reproduced.
- PF range RMSE below 0.5 m, about 0.08 m: below 0.5 m holds (0.23 m), the 0.08 m level does not. Ours sits at the d_hat noise level (std 0.245 m), so the echo vector adds little range information. Possible causes, untested: Q2 entries are standard deviations rather than variances (std 0.06 m would give roughly the paper's level), or the paper's range measurement differs from d + noise.
- EKF and UKF much worse than PF, and worse as K grows: not reproduced. Ours are about equal to the PF and nearly flat in K. In our model each car is tracked on its own and K only lowers the power per car; with a 310-element array the echo stays strong. The paper does not say what makes the EKF degrade with K (inter-vehicle interference and data association are both excluded by Sec. II-C and Remark 1).
- Computation order (EKF fastest, PF2000 slowest): reproduced. Absolute times are not: ours are 100 to 1000 times larger, because each measurement is a 310-element complex vector (Eq. 9) and the code is Python. "PF500 below 2 ms at K = 20" is not reproduced (ours 194 ms).

## Eq. (10) + noise vs exact straight-line tracks (angle deg^2)

Over 2 s, the noisy Eq. (10) truth differs from the straight-line track of the same car by RMS 4.7-5.7 deg in angle, 5.5 m in distance and 6.2 m/s in speed (mean over cars, all K). Without noise the drift is only 0.4 m and 1.8 deg (Stage 2), so nearly all of the difference comes from Q1: a speed noise of 1 m/s per 20 ms slot adds up to about 10 m/s of spread after 2 s. Because the noise is a random walk, some tracks become unphysical: of the 500 RSU-vehicle tracks (all K, 5 seeds), 21 reach a negative speed (lowest -7.9 m/s, i.e. driving backwards) and 23 come within 5 m of an RSU (closest 1.1 m). We keep them, since this is what Eq. (10) with Table III noise produces. They are a further hint that the Q1 values were not meant for this scale.
