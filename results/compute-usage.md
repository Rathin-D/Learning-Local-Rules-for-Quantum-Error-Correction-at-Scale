# Measured GPU usage on SubMIT, 2026-10-01 to 2026-11-16. MEASURED.
Command (on SubMIT; every GPU job script requests exactly 1 GPU, so GPU-hours = run time of jobs on GPU partitions;
SubMIT does not record gres in AllocTRES):
    sacct -u rathin -S 2026-10-01 -X -n -P --format=Partition,JobName,ElapsedRaw | awk -F'|' '$1 ~ /gpu/ {h[$2]+=$3/3600} ...
Total: 160.0 GPU-hours (A30 / RTX 6000) in 379 jobs. Not included: CPU-partition jobs (lake1, field1, online, pbbest, ...)
and all training/evaluation on the Macs (most model training, incl. v10-big and the seed lines).

| job | GPU-hours | what |
|---|---|---|
| qec-evalg | 94.2 | large-L evaluation wave w1b (v10-big, L up to 155, 378 pieces) |
| qec-dist | 22.1 | shadow-teacher distillation (student) |
| qec-train | 10.9 | K9 + K6 control, 48k steps each |
| qec-dg1 / dg1r / dg1p / dg2p | 10.7 | large-L diagnostics |
| qec-pBeval / pBtrain / pCtrain / pCeval | 13.0 | patch and circuit-noise training + evaluation |
| qec-sym2 | 4.2 | two 12k-step fine-tunes |
| others | 4.9 | probes, smoke tests, mech1, replay1, audit1, gate 1 |

Unit costs (A30-class): fine-tune 12k steps ~2 h; training 48k steps ~5.5 h; distillation ~22 h; one full large-L
evaluation wave to L155 ~94 h.
