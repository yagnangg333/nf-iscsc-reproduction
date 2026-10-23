# 05 Parameters (Stage 0)

Source: Yang et al., IEEE TVT vol. 75 no. 9, Sept 2026, pp. 20992-21008. "T-III" means Table III (p. 21003). Page numbers are the journal page numbers.

Values marked "not given" are gaps. They are discussed in 06-assumptions.md.

## Rayleigh distance check

Half-wavelength ULA, Nt = 310 (T-III), fc = 50 GHz (T-III).

- lambda = c / fc = 2.998e8 / 50e9 = 5.996 mm, so d_t = lambda/2 = 2.998 mm.
- Brief's formula: Nt^2 * lambda / 2 = 310^2 * 5.996e-3 / 2 = **288.1 m**.
- Using the true aperture D = (Nt - 1) d_t = 0.926 m: 2 D^2 / lambda = **286.2 m**.
- The paper's own example (Sec. I and Fig. 2 caption: D = 1 m at 50 GHz) gives 2 D^2 fc / c = 333.6 m, which the paper rounds to "approximately 300 m".

So the paper's Nt and fc give about 288 m. That is within 4% of "about 300 m", so the claim holds. Exactly 300 m would need Nt of about 316. The road segment (100 m long) sits well inside this region, so every vehicle is in the near field as the paper assumes (Sec. II).

## Table

| Symbol | Meaning | Value | Unit | Where |
|---|---|---|---|---|
| fc (written f) | Carrier frequency | 50 | GHz | T-III; Sec. I; Fig. 2 caption |
| lambda | Wavelength | 5.996 (derived c/fc) | mm | Derived |
| Nt | RSU transmit antennas (ULA) | 310 | - | T-III; Sec. II |
| Nr | Vehicle receive antennas | 3 (1 for the MISO comparison) | - | T-III; Sec. VI-B |
| d_t, d_r | Antenna spacing | lambda/2 | m | Sec. VI ("half-wavelength spaced"); Eq. (5) |
| M | Number of RSUs | 2 | - | T-III |
| K | Number of vehicles | 5 (default); 5 to 20 in Figs. 3, 5-7 | - | T-III; Figs. 3, 5, 6, 7 |
| Delta T | Time slot length | 0.02 | s | T-III; Eq. (10) |
| Pt | Total RSU power budget | 25 (= 0.316 W) | dBm | T-III; Eq. (34e) |
| sigma_c^2 | Communication noise power | -30 (= 1e-6 W) | dBm | T-III; Eq. (4) |
| sigma_r^2 | Echo noise power | -30 | dBm | T-III; Eq. (7) |
| B | Bandwidth | not given | Hz | - |
| iota | Word-to-bit conversion factor | 1.1 | - | T-III; Eq. (13) |
| rho | Semantic extraction ratio | 0.81 (role unclear, see gaps) | - | T-III; Eqs. (13), (14), (20), (34i) |
| Q, w_g, p_g, G | BLEU terms for rho lower bound | not given | - | Eq. (14) |
| F | Computing power coefficient | 10 (unit not given) | - | T-III; Eq. (20) |
| Q1 | Process noise covariance diag(sigma_theta^2, sigma_d^2, sigma_v^2, sigma_beta^2) | [0.02, 0.3, 1, 0.1] (units not given) | - | T-III; Eq. (12) |
| Q2 | Measurement noise covariance diag(sigma_1^2 I, sigma_2^2, sigma_3^2) | [0.04, 0.06, 1] (units not given) | - | T-III; Eq. (12) |
| C_mki | CPU cycles per bit | [1, 2] x 10^3 | cycles/bit | T-III; Eq. (22) |
| D_mki | Task data size | [1, 3] x 10^3 | bits | T-III; Eq. (22) |
| D (Figs. 8, 9) | Data sizes for computing figures | 1, 10, 100 | Kb | Figs. 8, 9 legends |
| T^max | Maximum processing latency | 0.015 | s | T-III; Eq. (34g) |
| F^max | Maximum RSU CPU frequency | 5.8 | GHz | T-III; Eq. (34h) |
| kappa | CPU energy-efficiency coefficient | 1e-28 | - | T-III; Eq. (25) |
| nu1 | Workload per degree of angle RCRB | 243.2 | "MHz/degree" | T-III; Eq. (24) |
| nu2 | Workload per metre of distance RCRB | 121.6 | "MHz/m" | T-III; Eq. (24) |
| nu3 | Extra workload term | ~ CN(0, 1000) | - | T-III; Eq. (24) |
| f1, f2 | Maps from RCRB to workload | linear ("multi-variable linear equation"); an exponential version also appears in Fig. 8, form not given | - | Sec. VI; Fig. 8 |
| epsilon (curly) | Objective weight, rate vs CRB | 0.5 | - | T-III; Eq. (34a) |
| epsilon (straight) | Convergence tolerance for Algorithm 3 | 0.1 | - | T-III; Alg. 3 line 7 |
| T (SA) | SA initial temperature | 100 | - | Alg. 2 line 11 |
| T_min, alpha, n_max | SA minimum temperature, cooling rate, max iterations | not given | - | Alg. 2 line 11 |
| Ns | Particle count | 500 and 2000 | - | Sec. VI-A; Fig. 3 |
| Resampling | PF resampling scheme | multinomial, every slot | - | Alg. 1 line 9 |
| T (FIM) | Number of samples in FIM | not given | - | Eqs. (16)-(18) |
| sigma_z^2 | Noise in FIM | not given (probably sigma_r^2) | - | Eqs. (16)-(18) |
| Road | Highway segment | 100 x 10 | m | Sec. VI |
| Lanes | Road layout | double-lane, straight, parallel to RSUs | - | Sec. II |
| Vehicle size | Length x width | 4.5 x 2 | m | Sec. VI |
| Min spacing | Minimum distance between vehicles | 3.5 | m | Sec. VI |
| Placement | Vehicle positions | "follow a Poisson distribution"; density not given | - | Sec. VI |
| v | Vehicle speed | not given (constant, same direction) | m/s | Sec. II; Eq. (10) |
| RSU positions | Where the 2 RSUs stand | not given ("either side of the road") | m | Sec. II |
| Simulation length | Tracking horizon | 2 s (= 100 slots) in Fig. 4 | s | Fig. 4 x-axis |

## Two numbers that do not fit together

These are worth knowing before Stage 5.

1. Eq. (20) with F = 10 and rho = 0.81 gives P_Comp = -10 ln(0.81) = 2.1 per served vehicle. If that is in watts, it alone is 6.7 times Pt (0.316 W), so (34e) can never hold. F must be in some smaller unit (mW would make it 2.1 mW).
2. Eq. (25) with kappa = 1e-28, C = 2000 and f = 5.8 GHz gives P_DT = 39 kW. Even the smallest CPU frequency that meets T^max for C = 2000, D = 3000 bits (0.4 GHz) gives 12.8 W. Fig. 9 shows DT power of 0.001 to 10 mW, and its curves grow with data size, but Eq. (25) has no data size term. Read by eye, Fig. 9 grows roughly as f^2 times D, not f^3 as in (25).
