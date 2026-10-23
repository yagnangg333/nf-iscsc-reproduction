# Open "your call" decisions (after Stage 1)

Gap numbers refer to 06-assumptions.md. Every other gap uses the suggested value.

| Gap | Question | My recommendation | Effect on results |
|---|---|---|---|
| 5 | Fix K per run, or draw K from a Poisson density? | Fix K (5, 10, 15, 20) and place vehicles uniformly, which is a Poisson process given its count | Low. Random K would blur the per-K curves of Figs. 3 and 5-7 |
| 10 | Main model: Eq. (5) Fresnel form, or exact spherical phase? | Eq. (5), as the paper writes it | Negligible on this road. Stage 1 shows the two match to correlation > 0.997 beyond 10 m; the nearest lane is 7.5 m away |
| 14 | Angle entries of Q1/Q2: degree^2 or rad^2? | degree^2, and run both in Stage 3 | High for Fig. 3(a). rad^2 makes the angle noise 57 times larger in standard deviation |
| 21 | Role of rho = 0.81 | Lower bound rho_LB = 0.81, starting rho = 1 | High for rate. It caps the semantic gain factor iota/rho at 1.36 |
| 26 | Exponential workload model in Fig. 8 (form not given) | Skip it and say so in the report | Only Fig. 8's dotted curves |
| 31 | Definition of "HH non-semantic" | rho = 1, iota = 1, no extraction power (plain Shannon rate) | High. It sets the size of the 18% / 20% gap directly; iota alone is 10% |

New from Stage 1 (gap 1, RSU position): with the RSU at the road end, only 14% of road points are materially near-field; at mid-road it is 34%. We keep the road-end placement as agreed, but it changes how much near-field modelling matters, so it goes in the report.
