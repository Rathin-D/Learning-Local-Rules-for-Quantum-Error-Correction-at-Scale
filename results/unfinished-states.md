# Mechanism study (PLAN-mechanism.md, revised). P3: orientation of the pairs still unfinished after 6L (p3_axis.py, Sun 10:20)
Saved diag1 states of v10-big (model_last), no new simulation. near-axis = k / D < 0.1 (k = smaller of |dx|, |dy|; minimum image).
Null: every lattice offset at the same distance equally likely (18.75-20%). No file hit the 50-state cap.
| L, p | saved | exactly 2 defects | pair distance D (median, range) | near-axis observed / expected | P(X >= observed) |
|---|---|---|---|---|---|
| 111, 1.6% | 9 | 9 | 47 (30-73) | 1 / 1.5 | 0.82 |
| 111, 1.7% | 111 | 102 (8 with 4, 1 with 6) | 49 (12-96) | 36 / 15.6 | 2.1e-7 |
| 155, 1.6% | 34 | 25 (9 with 4) | 66 (43-112) | 1 / 3.9 | 0.99 |
| pooled | | 136 | | 38 / 21.1 | 1.0e-4 (pass rule p < 0.01: MET) |
Joined with the original-memory replay (+40L), pairs at comparable distance (D within 25% of the median):
