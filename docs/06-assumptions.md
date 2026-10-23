# 06 Assumptions and gaps (Stage 0)

Each item says what the paper leaves open, the value I suggest, and why. Nothing here is in code yet. When we write config.py, each of these becomes a line marked ASSUMED. Items marked "your call" change the results enough that you should pick.

## Geometry and scenario (Stages 1-2)

1. **RSU positions.** Not given; Sec. II says only "along either side of the road", parallel to it. Suggest road x in [0, 100] m, y in [0, 10] m, RSU 1 at (0, -5) m and RSU 2 at (100, 15) m, arrays along x. Why: matches Fig. 2 (one on each side), keeps the array parallel to the road as Eq. (10) needs, and the 5 m setback keeps theta away from 0 where d sin(theta) terms vanish.
2. **Angle reference.** Taken as measured from the array axis (the road direction). Why: Eq. (5) puts cos(theta) on the along-array phase, and Eq. (10) shortens d by v dT cos(theta), which only fits that convention.
3. **Antenna index sets.** n_t from -(Nt-1)/2 to (Nt-1)/2 (same for n_r). Why: Sec. II-B says d and theta are measured from the array centres. With Nt = 310 (even) the indices are half-integers.
4. **Lanes.** Two lanes of 5 m, centres at y = 2.5 m and 7.5 m. Why: "double-lane" road, 10 m wide.
5. **Poisson placement.** Density not given, but every figure fixes K (5, 10, 15, 20). Suggest: for a given K, draw K positions uniformly along the 100 m (a Poisson process conditioned on its count is uniform), pick each lane at random, and redraw any vehicle that breaks the spacing rule. Your call if the task description asks for a random K instead (then we need a density lambda, e.g. K/100 per metre).
6. **Meaning of "minimum distance 3.5 m".** Suggest a 3.5 m gap between bumpers in the same lane, so centres are at least 8 m apart. Why: a 3.5 m centre-to-centre rule would let 4.5 m cars overlap. Check: 2 lanes x 12 cars of 8 m fit in 100 m, so K = 20 is possible but tight.
7. **Vehicle speed.** Not given. Suggest all vehicles at 20 m/s (72 km/h), same direction. Why: Sec. II says constant velocity, same direction, highway; one shared speed avoids cars in a lane running into each other.
8. **Simulation length.** 100 slots (2 s at dT = 0.02 s). Why: Fig. 4 runs 0 to 2 s.

## Channel model (Stage 1)

9. **Matrix sizes in Eqs. (4) and (13) do not match.** Gamma is Nr x Nt (Eq. 6) and A^H is Nr x Nt, so "Gamma A^H" cannot be an ordinary product. Suggest element-wise: H = (Gamma ⊙ A^H)^H, size Nt x Nr. Why: it is the only reading where H^H W H (Nr x Nr) in Eq. (13) works.
10. **Fresnel vs exact phase.** Eq. (5) (from [3, Eq. 30]) uses the second-order Fresnel approximation, while the path loss in Eq. (6) uses the exact distance. We implement (5) exactly as written for the model, and add the exact spherical phase only as a Stage 1 comparison. Your setup guide says "exact spherical distance" for Stage 1; the paper does not use that.
11. **Path loss has no wavelength.** Eq. (6) gives about -45 dB at 50 m, while free-space loss at 50 GHz over 50 m is about 100 dB. We implement (6) as written and note it. This makes SNRs far higher than a physical link.
12. **Sensing echo uses the vehicle's Nr antennas.** Eqs. (7) and (9) use A A^H with A of size Nt x Nr. A normal monostatic radar would use an Nt x Nt steering product. We implement as written and note it.
13. **Log base in Eq. (13).** Assume log2, since Fig. 5 is in bps/Hz.

## Tracking (Stage 3)

14. **Units of Q1 and Q2.** Table III gives Q1 = [0.02, 0.3, 1, 0.1] and Q2 = [0.04, 0.06, 1] with no units. If the angle entries are rad^2, the angle jumps about 8 degrees per 20 ms slot, which is far from the 1-9 degree RMSE in Fig. 3(a). If they are degree^2, the step is 0.14 degrees. Suggest: variances, angle entries in degree^2 (converted to rad^2 in code), others in SI. Your call; we can run both in Stage 3 and see which matches Fig. 3.
15. **What the filters measure.** Eq. (9) makes the angle measurement a full Nt-length complex echo vector; Eq. (11)-(12) then use sigma_1^2 I for it. Option A, as written: the filters use that vector (heavy, especially for EKF and UKF). Option B: the filters see a noisy angle directly. Suggest A, because the brief says "exactly as written", with B kept as a cross-check.
16. **Transmitted signal x_m,k,i inside Eq. (9).** Not specified for the tracking stage. Suggest a beam pointed at the predicted (theta, d) with Pt shared equally over the K vehicles.
17. **Initial beta.** Suggest beta_0 = 1 / (4 pi d_0^2), the round-trip version of Eq. (6) at the array centre. Why: keeps beta consistent with the channel model.
18. **Filter start and tuning.** Initial prior: Gaussian around the true state with covariance Q1. UKF: alpha = 1e-3, beta = 2, kappa = 0 (the standard Wan-van der Merwe values; [55] may give others). PF: multinomial resampling every slot, as Alg. 1 says.

## CRB (Stage 4)

19. **Missing real part in Eq. (16).** A Fisher matrix must be real, but (16) has no Re{}. Suggest J_ab = (2T|beta|^2 / sigma_z^2) Re Tr(dB_b Rx dB_a^H), the standard MIMO-radar form (Eq. 17 does carry Re{}). Also B in Eq. (17) is written with "s_m,k,i" where d_m,k,i is meant.
20. **T and sigma_z^2 in Eqs. (16)-(18).** Not given. Suggest sigma_z^2 = sigma_r^2 = -30 dBm and T = 1 sample. RCRB scales as 1/sqrt(T), so T only shifts the curves; we report it.

## Computing and power (Stage 5)

21. **Role of rho = 0.81.** The BLEU values Q, w_g, p_g needed for Eq. (14) are not given. Options: (a) rho_LB = 0.81 in (34i); (b) the starting rho^(0) in Alg. 3; (c) a fixed rho. Suggest (a), with rho^(0) = 1. Your call.
22. **Unit of F (Eq. 20).** With F = 10 W, extraction power alone is 2.1 W per vehicle, 6.7 times Pt. Suggest F in mW. Why: it is the only reading where (34e) can be met.
23. **Eq. (25) power for the DT.** With Table III values it gives watts to kilowatts, against Pt = 0.316 W, and it has no data-size term though Fig. 9 depends on data size. Options: (a) as written; (b) a per-slot energy form kappa f^2 C D / dT; (c) another scaling. Suggest we implement (a) first, check feasibility in Stage 5, and then decide together.
24. **nu1, nu2 units.** Read 243.2 MHz/degree as 243.2e6 CPU cycles per degree of RCRB(theta), and 121.6 MHz/m as 121.6e6 cycles per metre of RCRB(d).
25. **nu3 ~ CN(0, 1000).** A workload cannot be complex. Suggest |nu3| drawn per slot. It is about 30 cycles, so it barely matters.
26. **f1, f2.** Linear (f(x) = x), as Sec. VI says. The "exponential model" in Fig. 8 has no stated form, so Fig. 8's dotted curves cannot be reproduced without guessing. Your call: skip them, or try f(x) = exp(x) and label it ours.
27. **C and D draws.** Uniform on [1000, 2000] cycles/bit and [1000, 3000] bits, per vehicle.

## Assignment and optimisation (Stages 5-6)

28. **SA settings.** Only T0 = 100 is given. Suggest alpha = 0.95, T_min = 1e-3, n_max = 200, neighbour = move one random vehicle to the other RSU. Why: standard geometric cooling; 100 to 1e-3 at 0.95 takes about 225 steps, so n_max is the binding limit.
29. **SA direction.** (41a) is minimised, but Alg. 2 keeps a candidate when B_new >= B_n and starts B_best = 0, which is maximising. Suggest using the negative of (41a) as the SA score so both agree.
30. **"Greedy with random flip".** Number of flips not given. Suggest each vehicle switches RSU with probability 0.2, once. This is arbitrary; we will say so.
31. **"Non-semantic HH".** Not defined. Suggest rho = 1, iota = 1 and no extraction power, i.e. a plain Shannon rate. Your call. This choice drives the 20% result: iota/rho alone is 1.1/0.81 = 1.36.
32. **Conventional ISAC baseline ("Benchmark with Nr = 1", from [18]).** The paper does not describe its algorithm. I need [18] (Wang, Mu, Liu, "Near-field integrated sensing and communications") to implement it faithfully.
33. **Gaussian randomisation.** Number of draws not given. Suggest 100.
34. **Solver.** Problem (41) is an SDP with 310 x 310 complex matrices W and R per vehicle per RSU. That needs CVXPY (with SCS, or MOSEK under a free academic licence). CVXPY is not in your allowed library list. Your call. Run time per solve could be minutes, and SA calls it many times, so in Stage 5 we may have to optimise only some slots per seed.
35. **Monte Carlo runs.** The paper does not say how many runs each figure averages. We use your rule: at least 5 seeds, mean and standard deviation.
36. **Bandwidth.** Not given. Rates are in bps/Hz and Q2 is given directly, so we do not need it unless the comparison framework asks for bit/s.
