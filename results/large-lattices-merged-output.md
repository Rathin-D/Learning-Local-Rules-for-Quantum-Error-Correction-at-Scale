# Verbatim output of `python submit/submit_eval.py merge --tasks submit/tasks-w1b.txt --out /ceph/submit/data/user/r/rathin/qec-results`
# run on SubMIT Sat 2026-11-14 07:25 EST, while the array 7328552 was still running (PARTIAL: 313 of 378 shards).
# SubMIT tests, tag w1b (SIMULATED). Failures / runs after c*L noiseless rounds.

strict = logical error or defects left; left = runs with defects left; matched = logical error after a perfect matching finishes what is left (protection).

| model | L | p | runs (shards) | strict 2L | strict 6L | strict 20L | left 2L | left 6L | left 20L | matched 2L | matched 20L | reopened 2->6L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v10-big [model_last.pt] | 27 | 1.65% | 30000 (3/3) | 64 (2.1e-03) | 64 (2.1e-03) | 64 (2.1e-03) | 0 | 0 | 0 | 64 (2.1e-03) | 64 (2.1e-03) | 0 |
| v10-big [model_last.pt] | 39 | 1.65% | 30000 (3/3) | 51 (1.7e-03) | 51 (1.7e-03) | 51 (1.7e-03) | 0 | 0 | 0 | 51 (1.7e-03) | 51 (1.7e-03) | 0 |
| v10-big [model_last.pt] | 55 | 1.65% | 30000 (3/3) | 43 (1.4e-03) | 39 (1.3e-03) | 39 (1.3e-03) | 6 | 1 | 0 | 38 (1.3e-03) | 39 (1.3e-03) | 0 |
| v10-big [model_last.pt] | 79 | 1.60% | 100000 (20/20) | 304 (3.0e-03) | 278 (2.8e-03) | 244 (2.4e-03) | 293 | 256 | 211 | 52 (5.2e-04) | 57 (5.7e-04) | 0 |
| v10-big [model_last.pt] | 79 | 1.65% | 100000 (20/20) | 623 (6.2e-03) | 524 (5.2e-03) | 476 (4.8e-03) | 592 | 470 | 405 | 114 (1.1e-03) | 123 (1.2e-03) | 0 |
| v10-big [model_last.pt] | 79 | 1.70% | 50000 (10/10) | 546 (1.1e-02) | 455 (9.1e-03) | 409 (8.2e-03) | 523 | 403 | 342 | 108 (2.2e-03) | 120 (2.4e-03) | 0 |
| v10-big [model_last.pt] | 79 | 1.80% | 20000 (4/4) | 846 (4.2e-02) | 745 (3.7e-02) | 687 (3.4e-02) | 777 | 621 | 538 | 274 (1.4e-02) | 279 (1.4e-02) | 0 |
| v10-big [model_last.pt] | 111 | 1.60% | 100000 (50/50) | 1113 (1.1e-02) | 757 (7.6e-03) | 521 (5.2e-03) | 1112 | 741 | 451 | 63 (6.3e-04) | 122 (1.2e-03) | 0 |
| v10-big [model_last.pt] | 111 | 1.65% | 100000 (50/50) | 2233 (2.2e-02) | 1515 (1.5e-02) | 1099 (1.1e-02) | 2223 | 1478 | 956 | 158 (1.6e-03) | 310 (3.1e-03) | 0 |
| v10-big [model_last.pt] | 111 | 1.70% | 50000 (25/25) | 2054 (4.1e-02) | 1461 (2.9e-02) | 1103 (2.2e-02) | 2044 | 1421 | 959 | 212 (4.2e-03) | 340 (6.8e-03) | 0 |
| v10-big [model_last.pt] | 111 | 1.80% | 20000 (10/10) | 2859 (1.4e-01) | 2208 (1.1e-01) | 1756 (8.8e-02) | 2795 | 2071 | 1432 | 599 (3.0e-02) | 763 (3.8e-02) | 0 |
| v10-big [model_last.pt] | 155 | 1.60% | 50000 (100/100) | 2344 (4.7e-02) | 1797 (3.6e-02) | 1371 (2.7e-02) | 2343 | 1795 | 1354 | 31 (6.2e-04) | 217 (4.3e-03) | 0 |
| v10-big [model_last.pt] | 155 | 1.70% | 7500 (15/60) | 1152 (1.5e-01) | 912 (1.2e-01) | 730 (9.7e-02) | 1152 | 911 | 714 | 60 (8.0e-03) | 157 (2.1e-02) | 0 |

missing shards: 65 of 378
