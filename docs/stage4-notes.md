# Stage 4 notes: Cramer-Rao bound

## The paper's expressions (Sec. III-B), quoted

- Parameters: Psi = [d, theta, beta] (distance, angle, round-trip path loss).
- Eq. (15): J = [[J_dd, J_dtheta, J_dbeta], [J_thetad, J_thetatheta, J_thetabeta], [J_betad, J_betatheta, J_betabeta]] = [[J11, J12], [J12^T, J22]].
- Eq. (16): J_ab = (2T|beta|^2 / sigma_z^2) Tr(dB_b Rx dB_a^H), for a, b in {theta, d}.
- Eq. (17): J_abeta = (2T beta^* / sigma_z^2) Re{Tr(B Rx dB_a^H)} [1 j].
- Eq. (18): J_betabeta = (2T / sigma_z^2) Tr(B Rx B^H) I.
- B = A(theta, d) A^H(theta, s) (the paper prints "s"; it must be d), dB_theta = dB/dtheta, dB_r = dB/dd.
- Eq. (19): CRB(d) = J^-1[1,1], CRB(theta) = J^-1[2,2], with J^-1 = (J11 - J12 J22^-1 J12^T)^-1.

## Derivation, step by step

1. Signal model. After the matched filter (Eqs. 7-9, Remark 1), the echo from one vehicle over T snapshots is
   z(t) = beta B x(t) + n(t), t = 1..T, with n(t) ~ CN(0, sigma_z^2 I) and B = A A^H (Nt x Nt).
   The mean is mu(t) = beta B(theta, d) x(t); the noise does not depend on the parameters.

2. Real parameters. beta is complex, so we split it: xi = [d, theta, Re beta, Im beta] (four real numbers). That is why Eqs. (17)-(18) carry [1 j] and I (2 x 2).

3. FIM for a complex Gaussian with known covariance sigma^2 I (the Slepian-Bangs formula):
   J_ij = (2 / sigma^2) Re sum_t (d mu(t)/d xi_i)^H (d mu(t)/d xi_j).

4. The four derivatives of mu(t):
   d mu/d d = beta dB_d x(t), d mu/d theta = beta dB_theta x(t), d mu/d Re beta = B x(t), d mu/d Im beta = j B x(t).

5. Sum over snapshots. With Rx = (1/T) sum_t x(t) x(t)^H (Eq. 3), any term sum_t (P x(t))^H (Q x(t)) equals T Tr(Q Rx P^H).

6. Put 4 and 5 into 3:
   - angle/distance block: J_ab = (2T/sigma^2) Re{ |beta|^2 Tr(dB_b Rx dB_a^H) }. This is Eq. (16) with Re{} added. Without Re{} the entry can be complex and J is not a valid FIM, so Re{} must be there (gap 19).
   - cross block: J_a,Re beta = (2T/sigma^2) Re{ beta^* Tr(B Rx dB_a^H) } and J_a,Im beta = (2T/sigma^2) Re{ j beta^* Tr(B Rx dB_a^H) }. Together: (2T/sigma^2) Re{ beta^* Tr(B Rx dB_a^H) [1, j] }. The paper puts beta^* outside Re{}, which would make the entry complex; we keep it inside (same reading as the papers this form comes from).
   - beta block: J_Re,Re = J_Im,Im = (2T/sigma^2) Tr(B Rx B^H), J_Re,Im = Re{ j Tr(...) } = 0, so the block is (2T/sigma^2) Tr(B Rx B^H) I. This is Eq. (18) exactly.

7. Nuisance parameter. beta is unknown but not of interest, so it is removed with the Schur complement: Jbar = J11 - J12 J22^-1 J12^T (2 x 2, for d and theta), and CRB(d) = Jbar^-1[1,1], CRB(theta) = Jbar^-1[2,2]. This is Eq. (19).

8. Derivatives of B. From Eq. (5), A^H[r, t] = exp(-j k psi_rt) with
   psi_rt = p(delta_r) - p(delta_t) + delta_r delta_t sin^2(theta) / d, p(delta) = -delta cos(theta) + delta^2 sin^2(theta) / (2d), k = 2 pi / lambda.
   So dA^H/da = -j k (d psi/da) A^H, and dB = (dA^H)^H A^H + A dA^H. In code we apply these to the beams (B x) and never build the 310 x 310 matrix.

## Choices in the code

- sigma_z^2 = sigma_r^2 = -30 dBm and T = 1 (gap 20, accepted).
- beta = 1 / (4 pi d), the same scale as Stage 3.
- Rx is passed as beams X with Rx = X X^H. Before Stage 5 there is no optimised beamformer, so A3 uses one equal-power probe beam per vehicle (Pt/K each) and Rx includes every beam (Eq. 3).

## Why a second bound for the filters

The guide asks that filter RMSE never fall below the root CRB. The paper's CRB (and any one-snapshot CRB) only bounds an estimator that uses one slot of data and no prior. The Stage 3 filters also use the motion model (Eq. 10), all earlier slots and d_hat, v_hat, so they can legitimately beat a one-snapshot CRB. The correct check is the posterior CRB (Tichavsky, Muravchik and Nehorai, IEEE Trans. Signal Process., 1998), which uses the same state and measurement models as the filters. We compute both and test RMSE against the posterior CRB.

Approximations in the posterior CRB: the expectations are averaged over 10 Monte Carlo tracks per vehicle, and with "relative" beta noise the dependence of Q on beta is ignored (Q is evaluated at the true state).

## Results

All numbers from `python experiments/stage4_crb.py` (5 seeds where random). Files in results/stage4_*.

### A. The paper's CRB (Eqs. 15-19)

- Check of the code: a Monte Carlo maximum-likelihood estimator (Part C, 5 seeds) gives MSE/CRB = 0.99 +/- 0.12 for angle alone, and 1.10 +/- 0.10 (angle) and 1.06 +/- 0.10 (distance) when angle, distance and beta are estimated jointly. An efficient estimator should sit at 1, so the bound is computed correctly.
- One vehicle, all power Pt in one beam, -30 dBm noise (A1): the root CRB grows in proportion to distance for both angle and distance (log-log slope 1 in stage4_crb_vs_distance.png). At d = 30 m, theta = 60 deg: 1.2e-4 deg and 0.018 m.
- Against transmit SNR at d = 50 m (A2): 10 dB more SNR gives sqrt(10) less error, as it must. 0.0036 deg at 30 dB, 1.1e-4 deg at 60 dB.
- On the road with one equal-power beam per vehicle (A3, no optimisation yet): angle 0.0018-0.0023 deg (median), distance 3.1-4.3 m (median; mean 17-21 m because of a long tail). 28% of vehicle-RSU pairs are under 0.5 m. Nearly flat in K, because Rx (Eq. 3) adds up every beam, so the total power on the road stays Pt.

Paper (by eye): Fig. 6 HH angle RCRB is about 1e-3 to 1e-2 deg, so our angle numbers are in the same range. Fig. 7 distance RCRB is about 5e-3 to 0.5 m; ours is 3-4 m (median). Possible reasons: our beams are not optimised (Stage 5), and distance information in the near field comes from the curvature term delta^2 sin^2(theta) / (2d), which is weak for cars far down the road and close to the array axis. We do not adjust anything to close this gap.

### B. Do the Stage 3 filters respect a lower bound?

- One-snapshot CRB (the echo of one slot, plus d_hat and v_hat, no prior): median angle bound 5.1-8.5 deg. Its mean is meaningless (up to 1e6 deg) because cars near the array axis (sin(theta) near 0) give almost no angle information in one slot. The filters (about 1 deg) beat this easily, which is expected: they use the motion model and 100 slots. So this bound is not the right check.
- Posterior CRB (the right check): root bound 0.35-0.43 deg (angle), 0.224 m (distance), 0.785 m/s (velocity).
- Angle: every run of every filter is at least 1.8 times the bound (seed-pooled 2.4-3.6 times). Holds with room to spare.
- Distance and velocity: pooled over the 5 seeds, RMSE/bound is 1.006-2.0 (distance) and 0.996-1.21 (velocity). The EKF sits on the bound. 12 of 80 single runs (distance) and 14 of 80 (velocity) fall 0.1-4% under it, lowest 0.959.
- Why this is not a bug: velocity in Eq. (10) is a random walk with a direct measurement v_hat, a linear Gaussian problem where the Kalman filter is exactly optimal and its error equals the bound (we get 0.785 m/s from the bound recursion, and the closed form sqrt((sqrt(5) - 1)/2) = 0.786 m/s). One run has only 2K tracks, so its RMSE is a noisy estimate. A separate simulation of the optimal Kalman filter with 10 tracks of 100 slots gives RMSE/bound 0.947-1.052 (95% range); with 40 tracks 0.974-1.023. Our lowest values are inside these ranges, and the low runs are the same seeds for all filters (seed 4 for velocity, seed 3 for distance), so they come from that seed's noise draw, not from one filter. The 0.996 pooled value (EKF, K = 15) is within one standard error of 1.
- Approximations in the bound that could also move it by a little: 10 Monte Carlo tracks per vehicle for the expectations, and dQ/dbeta ignored for "relative" beta noise.

### Two noise levels in the paper

The paper's CRB uses sigma_z^2 = -30 dBm (1e-6 W). Table III gives the echo measurement noise for tracking as sigma1^2 = 0.04. Both describe noise on the echo, but they differ by 46 dB. With 0.04 the echo SNR of one slot is about 10 dB (K = 5); with -30 dBm it is about 56 dB. This is why the filters reach about 1 deg while the paper-CRB reaches 0.002 deg. We keep each value where the paper uses it, and record the mismatch. It also matters for Stage 5, where the CRB enters the computing load through nu1 and nu2.

### Verdict

- The CRB expressions of Eqs. (15)-(19), after the two corrections above, are reproduced and checked against an ML estimator.
- Fig. 6 angle RCRB range: matched in order of magnitude with unoptimised beams.
- Fig. 7 distance RCRB: not reproduced (ours about 10 times larger in the median). To be revisited after Stage 5 beamforming.
- Filter RMSE vs bound: holds for angle with room to spare; for distance and velocity the filters are at the bound, as theory predicts for a linear Gaussian part of the state.
