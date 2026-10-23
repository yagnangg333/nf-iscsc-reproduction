# Stage 0 notes: Yang et al. 2026

Parameters and the Rayleigh check are in 05-parameters.md. Gaps are in 06-assumptions.md.

## 1. System model in my own words

Two roadside units (RSUs) stand beside a straight two-lane highway. Each has a 310-element half-wavelength array at 50 GHz, so the near-field region reaches about 290 m and every vehicle on the 100 m stretch is in it. Each vehicle has 3 receive antennas. In every 20 ms slot each RSU sends one joint signal: communication beams carrying semantically compressed messages to the vehicles it serves, plus sensing beams. The echoes let the RSU track all vehicles (angle, distance, speed, path loss) with a particle filter, and it builds a digital twin from those tracks. Each vehicle is served by exactly one RSU, but it hears interference from every beam of both RSUs. The channel uses the near-field (spherical-wave, Fresnel-approximated) steering model, so beams focus on an angle and a distance. Each RSU has one power budget that must cover the beams, the neural semantic encoder (a smaller extraction ratio rho means fewer bits sent but more compute power), and the CPU work for the digital twin. That CPU work grows when sensing is worse, through the root CRB. The design problem picks the vehicle-to-RSU assignment, the beamformers, rho and the CPU frequencies to trade semantic rate against angle and distance CRB.

## 3. Equations to implement, by stage

**Stage 1, near-field channel.** Eq. (5): steering vectors a^T, a^R, cross-term H and A^H = a^R (a^T)^H ⊙ H. Eq. (6): element path loss Gamma and phi. Eq. (4): received signal (desired, communication interference, sensing interference). Far-field comparison: Eq. (5) with the d terms dropped (the paper has no separate far-field equation; that is ours).

**Stage 2, road scenario.** No equations; Sec. VI text (road, vehicle size, spacing, Poisson) and Sec. II (two lanes, constant speed, same direction).

**Stage 3, tracking.** Motion model Eq. (10). State/measurement form Eq. (11). Noise covariances Eq. (12) with Table III. Measurements: Eq. (9) for angle and beta; d_hat = c tau/2 + z_tau and v_hat = lambda mu/2 + z_mu (Sec. II-C text); Eqs. (7)-(8) describe how tau and mu come out of the matched filter, but Remark 1 lets us skip simulating the matched filter. PF: Eqs. (26)-(33) and Algorithm 1. EKF and UKF are not written out in the paper; they are cited ([2], [55]), so we use the standard textbook forms on the same Eqs. (10)-(12).

**Stage 4, CRB.** Eqs. (15)-(19): FIM blocks, Schur complement, CRB(d) = J^-1[1,1], CRB(theta) = J^-1[2,2]. Needs Rx from Eq. (3) and B = A A^H.

**Stage 5, compute, power and assignment.** Semantic rate Eq. (13) (and its log-det form Eq. 35). rho bound Eq. (14) (cannot be evaluated, see gap 21). Powers Eqs. (20), (21), (25). Latency Eqs. (22)-(24). Problem Eq. (34a)-(34i). SA assignment: Algorithm 2.

**Stage 5, beamforming.** Lemma 1 Eq. (36) and its use Eq. (37); Lemma 2 Eq. (38); convex rate constraint Eq. (39); CRB constraint via Schur complement Eq. (40); convex problem Eq. (41); Algorithm 3; Gaussian randomisation (Alg. 3 line 8). Whole loop: Algorithm 4.

**Stage 6, baseline.** "Benchmark with Nr = 1" comes from [18]; the paper gives no equations for it. "HH non-semantic" uses the same equations with semantic extraction switched off (gap 31).

## 4. Optimisation method for beamforming

The paper uses **alternating optimisation (AO)** with **semidefinite relaxation (SDR)** inside it, and a **greedy plus simulated annealing** search (with a tabu list) outside it for the vehicle-to-RSU assignment.

In plain words: the joint problem mixes a 0/1 assignment with continuous beamformers, and the rate and CRB terms are non-convex, so no solver handles it in one go. The outer loop (Algorithm 2) fixes who serves whom: start from "nearest RSU", then try random swaps, sometimes keeping a worse one early on so the search does not get stuck. For each assignment the inner loop (Algorithm 3) solves the rest. The beamformers are written as matrices W = w w^H and R = r r^H and the rank-one rule is dropped (that is SDR), which makes them convex variables. Three tricks then make each sub-problem convex. Lemma 1 (the WMMSE-style log-det identity) turns -log det into a linear term plus an auxiliary matrix A. Lemma 2 (a low-SNR approximation) turns the interference log-det into a trace. The Schur complement turns the CRB limit into a matrix inequality (40). AO then cycles: solve for W, R, f with A and rho fixed; update A; find rho by bisection; repeat until the rate changes by less than 0.1. Finally, Gaussian randomisation pulls rank-one beam vectors out of W and R. The paper's reason is that each step is convex, so each step can only improve the objective, which gives convergence to a stationary point (Appendix A). It is not a global optimum.

## 6. Figures and tables vs task objectives

Mapped against 02-task-description.txt (pasted 23 Oct). The comparison framework ("below" in the brief) has not arrived yet, so objective 6 is only partly mapped.

| Brief objective | Paper source | Our stage | Note |
|---|---|---|---|
| 1. System model: ULA, Nt half-wavelength, MU-MIMO, ~300 m near field; Poisson vehicles on 100 m x 10 m, 4.5 m x 2 m cars, 3.5 m spacing | Sec. II; Table III; Sec. VI text; Fig. 2 | 1, 2 | Rayleigh check gives 288 m. No figure for the road layout |
| 2. Near-field steering vector and channel; compare with far field; find where they diverge | Eqs. (4)-(6) | 1 | The paper has no near/far comparison (Sec. VII calls it future work). The divergence range will be our own result |
| 3. PF (500, 2000), EKF, UKF; angle, distance, velocity RMSE and time vs K | Fig. 3(a)-(d); Eqs. (9)-(12), (26)-(33), Alg. 1. Fig. 4 is extra (DT X/Y error) | 3 | Direct match |
| 4a. Resource allocation and beamforming with SA assignment, under CRB and rate constraints | Eqs. (13)-(25), (34)-(41); Algs. 2-4; Fig. 5 | 5 | |
| 4b. CRB for angle as sensing benchmark | Eqs. (15)-(19); Fig. 6 (Fig. 7 for distance) | 4 | |
| 5. 20% rate gain over conventional ISAC at equal sensing accuracy, at 1, 10 and 100 Kb | Abstract; Fig. 5 with Figs. 6-7; data sizes only in Figs. 8-9 | 6 | Paper never shows this at 1/10/100 Kb; see below |
| 6. Documented code, report, comparison framework measurements | Figs. 8-9 (CPU frequency and power vs data size) are the likely cost numbers for the Task 1 comparison | 7 | Needs 03-comparison-framework.pdf and Task 1 results |

**Objective 5 does not match one figure.** The 20% comes from Fig. 5 and the line under it: "With the HH algorithm, when semantic communication is not used, the average semantic rate drops by 18%". Read by eye, Fig. 5 gives HH 3.8 vs HH non-semantic 3.1 bps/Hz at K = 5 and 2.8 vs 2.3 at K = 20, about 22% higher with semantics. Figs. 6 and 7 show both at about the same RCRB, which is the "equal sensing accuracy" part. Against the other baseline in Fig. 5, the ISAC beamformer from [18] with Nr = 1, the gap by eye is about 14% at K = 5 and over 200% at K = 20. So "conventional ISAC" has two possible readings, and they give very different answers.

The 1, 10 and 100 Kb sizes appear only in Figs. 8 and 9. Running the Fig. 5 comparison at those sizes is our extension, not something the paper reports.

**100 Kb is infeasible with Table III values.** Latency (22) with T^max = 0.015 s needs f >= C D / T^max per vehicle. With C = 1000 cycles/bit (the lowest in Table III), D = 100 Kb needs 6.7 GHz for one vehicle, above F^max = 5.8 GHz for the whole RSU. At 10 Kb an RSU can serve 4 to 8 vehicles, so K = 20 over 2 RSUs also fails. At 1 Kb it fits easily. Unless "Kb" or C means something else, the 100 Kb case (and 10 Kb at high K) will come out "infeasible" rather than as a rate number. That itself is a result we can report.
