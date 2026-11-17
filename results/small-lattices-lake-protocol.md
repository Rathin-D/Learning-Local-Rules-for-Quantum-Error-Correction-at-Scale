# lake1: our rule measured like Lake's paper measures his (v10-big model_last.pt; SubMIT CPU array 7340954; Sat 20:23)
Toric code, bit flips p on data and readings each round, L noisy rounds, then up to 2L noiseless rounds of the rule.
Strict = defects left or logical error; Lake-style (unfinished runs scored by their winding) differs by at most 0.5% here.
Runs: 300k (L5-13), 200k (L19), 100k (L27, L39); 4 of 490 shards missing (L39 1.7%: 95k; L27 and L39 at 2.0%: 90k). SIMULATED.

Strict failure rate after 2L:
| p | L5 | L7 | L9 | L13 | L19 | L27 | L39 |
|---|---|---|---|---|---|---|---|
| 1.2% | 6.5e-3 | 2.5e-3 | 1.2e-3 | 2.6e-4 | 5.5e-5 | 3.0e-5 | 1.0e-5 |
| 1.4% | 1.1e-2 | 5.5e-3 | 2.8e-3 | 1.0e-3 | 3.6e-4 | 1.9e-4 | 7.0e-5 |
| 1.5% | 1.4e-2 | 7.3e-3 | 4.2e-3 | 1.8e-3 | 7.8e-4 | 4.2e-4 | 3.1e-4 |
| 1.6% | 1.7e-2 | 1.0e-2 | 6.6e-3 | 3.2e-3 | 1.8e-3 | 1.1e-3 | 1.0e-3 |
| 1.7% | 2.1e-2 | 1.4e-2 | 1.0e-2 | 6.2e-3 | 3.6e-3 | 3.1e-3 | 3.0e-3 |
| 1.8% | 2.6e-2 | 1.8e-2 | 1.4e-2 | 9.5e-3 | 7.5e-3 | 6.7e-3 | 8.6e-3 |
| 2.0% | 3.8e-2 | 3.1e-2 | 2.8e-2 | 2.6e-2 | 2.7e-2 | 3.6e-2 | 6.0e-2 |
Findings
1. Over Lake's own toric sizes (L = 5-19) our failure rate FALLS with size at every p up to 1.8%, and is flat from L9 to L19 at
   2.0%: in that size range the finite-size crossing is at ~1.9-2.0%. Lake reports ~1.5% (rounded to 0.5%) from L = 5-19.
2. Including L27 and L39, the crossing drifts down: flat from L19 to L39 at 1.7%, rising at L39 for 1.8% and from L13 on at 2.0%:
   crossing ~1.7-1.8% for L <= 39. Together with the large-L results (rise from L79 on at 1.6%) the crossing keeps drifting down
   with size: it is a finite-size (pseudo-)threshold, not an asymptotic one.
3. Essentially nothing is left unfinished up to L39 (at most 49 of 300,000), so Lake-style and strict agree.
4. To state the comparison exactly we still need Lake's noise model for his toric figure (our earlier reading: independent data and
   readout bit flips with equal p, as here) - to be checked against the paper before writing.

## Complete (rerun 7345618, Sun 16:37): all 490 shards in. The 3 points that were short now have full runs:
L39 1.7%: 297 / 100,000 = 3.0e-3 (was 286 / 95,000 = 3.0e-3); L27 2.0%: 3,626 / 100,000 = 3.6e-2 (was 3.6e-2 from 90k);
L39 2.0%: 5,972 / 100,000 = 6.0e-2 (was 6.0e-2 from 90k). L39 1.2% and 1.4% unchanged (their timed-out tasks had already
written their results). No change to any rate or to the findings above.
