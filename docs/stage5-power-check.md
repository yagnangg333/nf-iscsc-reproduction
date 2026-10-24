# Power constraint check (before Stage 5)

Nothing in the code has been changed yet. This note checks why the power budget (34e) cannot be met with the Table III values, and gives options.

## What the paper says

- (34e): P_C&S + P_Comp + sum over k of P_DT <= Pt, with Pt = 25 dBm = 316 mW (Table III).
- Eq. (20): P_Comp = -F sum_k xi ln(rho), with F = 10 and no unit. "F is a coefficient that converts a magnitude to its power." Cites [10], [38].
- Eq. (21): P_C&S = Tr(W W^H + R R^H), the beam power.
- Eq. (22)-(24): latency T = (C D + L) / f, with L = nu1 f1(RCRB) + nu2 f2(RCRB) + nu3.
- Eq. (25): P_DT = kappa f^3 C, kappa = 1e-28, C = 1000-2000 cycles/bit. Cites [44]-[46].
- Fig. 9: "Power consumed (mW)" against "maximum CPU frequency (GHz)" for data sizes 1, 10 and 100 Kb.

## Numbers with Table III as written

The smallest CPU frequency that meets T^max = 15 ms is f = C D / T^max (ignoring L for now).

| C (cycles/bit), D (bits) | f needed | Eq. (25) as written | kappa f^2 C D / dT | kappa f^3 |
|---|---|---|---|---|
| 2000, 3000 (worst case) | 0.40 GHz | 12.8 W | 4.8 mW | 6.4 mW |
| 1500, 2000 (middle) | 0.20 GHz | 1.2 W | 0.6 mW | 0.8 mW |
| 1000, 1000 (best case) | 0.07 GHz | 30 mW | 0.02 mW | 0.03 mW |
| any, at F^max = 5.8 GHz | 5.8 GHz | 20-39 kW | 170-1000 mW | 19.5 W |

Eq. (20) with F = 10 W and rho = 0.81 gives 2.1 W per vehicle, 6.7 times Pt on its own. With F in mW it is 2.1 mW per vehicle.

So, as written, a middle-sized task alone needs about 4 times Pt, and one vehicle's semantic extraction needs about 7 times Pt. Problem (41) would be infeasible for nearly every assignment, and Algorithm 2 would never find a feasible plan.

## Checking each likely cause

1. Unit mix-up in Table III (F). The paper gives no unit for F. In W the budget can never hold; in mW (or in dBm, since 10 dBm = 10 mW) it gives 2.1 mW per vehicle, which fits. Likely.
2. Eq. (25) is energy, not power. Fig. 9 grows exactly as f^2 (fitted log-log slope 2.00 for the 1 Kb curve: 0.001, 0.004, 0.009, 0.016, 0.025, 0.036 mW at 1-6 GHz) and in proportion to D (each curve is 10 times the one below for 10 times the data). Eq. (25) grows as f^3 and has no D. The common energy model in mobile edge computing papers is E = kappa f^2 C D (joules per task), which has exactly Fig. 9's shape. Likely. Fig. 9's scale matches kappa f^2 C D with C = 10 cycles/bit, which is outside Table III's 1000-2000, so Fig. 9 fixes the shape but not the constant.
3. Missing scaling factor. If (25) is energy, it needs dividing by a time to become power. The slot length dT = 20 ms (energy per slot = average power) is the natural choice; T^max = 15 ms is the other. With f set exactly to meet the latency, kappa f^2 C D / T^max = kappa f^3, the textbook CPU power. So causes 2 and 3 are the same fix.
4. A typo (f^3 for f^2, or C for C D). This is the same change as 2 seen from the other side. We cannot tell a typo from a modelling choice without [44]-[46].

A separate risk, not about power: L in Eq. (24). With our Stage 4 distance RCRB (about 3 m median, unoptimised beams), nu2 RCRB(d) = 3.6e8 cycles, which alone needs 24 GHz to meet 15 ms (F^max = 5.8 GHz). With the paper's Fig. 7 level (0.05 m) it needs 0.4 GHz. So the latency constraint (34g) will be met only if Stage 5 beamforming brings the distance CRB down near the paper's level. We will report this, not tune around it.

## Options

- A. As written. F in W, P_DT = kappa f^3 C. Expected result: infeasible almost always; Stage 5 and 6 produce no results. Useful only as a documented run.
- B. F in mW, and Eq. (25) read as energy per slot: P_DT = kappa f^2 C D / dT. Fits Fig. 9's shape (f^2, linear in D), all terms are a few mW, and the beams keep most of Pt. (Recommended.)
- C. F in mW, and the textbook CPU power P_DT = kappa f^3 (drop C). Feasible too, but it has no D term, so it cannot give Fig. 9's dependence on data size.
- D. F in mW, Eq. (25) as written. Feasible only for small tasks (C D below about 2e6); fails for the middle and worst cases above.

Whatever is chosen, every reading stays a switch in config.py, and the as-written run (A) is kept for the report, the same way as the beta noise in Stage 3.
