"""P3 of PLAN-mechanism.md (revised after review): are the far pairs left unfinished after 6L near an AXIS more often than
chance, and do near-axis pairs (at comparable distance) more often stay stuck or close the wrong way?
Uses only saved data from diag1 (states-score-*.pt and replay-*.json in qec-results/diag1). SIMULATED. CPU, seconds.

  python p3_axis.py check
  python p3_axis.py run --out /ceph/submit/data/user/r/rathin/qec-results
"""
import argparse
import hashlib
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decoder_v41 import synd  # noqa: E402

CODE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
NEAR = 0.1          # near-axis: k / D < 0.1, with k = min(|dx|, |dy|), D = |dx| + |dy| (minimum image on the torus)


def min_image(a, b, L):
    """signed minimum-image offset from a to b on an L x L torus (L odd here: unique)."""
    out = []
    for i in range(2):
        d = (b[i] - a[i]) % L
        if d > L // 2:
            d -= L
        out.append(d)
    return out


def null_fraction(L, D):
    """fraction of all minimum-image offsets (dx, dy) with |dx| + |dy| = D that are near-axis (each lattice site at that
    distance equally likely)."""
    h = L // 2
    tot = near = 0
    for dx in range(-h, h + 1):
        r = D - abs(dx)
        if r < 0 or r > h:
            continue
        for dy in ({r, -r} if r else {0}):
            tot += 1
            near += (min(abs(dx), abs(dy)) / D < NEAR)
    return near / tot if tot else float("nan"), tot


def poisson_binomial_tail(ps, x):
    """P(X >= x) for X = sum of independent Bernoulli(p_i) (exact, by recursion)."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[x:].sum()), float(np.dot(np.arange(len(dist)), dist))


def fisher_greater(a, b, c, d):
    """one-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]]: P(first row has >= a 'events')."""
    n1, n2, k = a + b, c + d, a + c
    tot = comb(n1 + n2, k)
    return sum(comb(n1, i) * comb(n2, k - i) for i in range(a, min(n1, k) + 1)) / tot


def analyse(states, L):
    rows = []
    for s in states:
        d = (s["e"] ^ s["corr"]).unsqueeze(0)
        pts = torch.nonzero(synd(d)[0]).tolist()
        row = dict(shot=s["shot"], seed=s.get("seed"), p=s.get("p"), ndef=len(pts))
        if len(pts) == 2:
            dx, dy = min_image(pts[0], pts[1], L)
            D = abs(dx) + abs(dy)
            row.update(dx=dx, dy=dy, D=D, k=min(abs(dx), abs(dy)), near=min(abs(dx), abs(dy)) / D < NEAR,
                       p0=null_fraction(L, D)[0])
        rows.append(row)
    return rows


def summarise(rows_by_key, replay_by_key):
    out = ["# P3: orientation of the pairs still unfinished after 6L (v10-big, diag1 saved states). SIMULATED.",
           f"near-axis = k / D < {NEAR}; null = all lattice offsets at the same distance equally likely (minimum image);"
           " pooled test Poisson-binomial. Then, joined with the original-memory replay (+40L): stuck / wrong-way by orientation.",
           ""]
    res = {}
    for key, rows in sorted(rows_by_key.items()):
        L, p = key
        n = len(rows)
        two = [r for r in rows if r["ndef"] == 2]
        x = sum(r["near"] for r in two)
        tail, mean = poisson_binomial_tail([r["p0"] for r in two], x) if two else (float("nan"), 0.0)
        Ds = [r["D"] for r in two]
        out += [f"## L = {L}, p = {100 * p:.2f}%: {n} saved states, {len(two)} with exactly 2 defects "
                f"({', '.join(f'{c} with {k}' for k, c in sorted(_count(r['ndef'] for r in rows).items()) if k != 2) or 'no others'})",
                f"distance D of the pair: median {np.median(Ds) if Ds else '-'}, range {min(Ds) if Ds else '-'}-{max(Ds) if Ds else '-'}",
                f"near-axis pairs: {x} of {len(two)} observed vs {mean:.1f} expected by chance; P(X >= {x}) = {tail:.2e}"]
        rep = replay_by_key.get(key, {})
        joined = [(r, rep.get((r["seed"], r["shot"]))) for r in two]
        joined = [(r, q) for r, q in joined if q is not None]
        res[key] = dict(n=n, two=len(two), near=x, expected=mean, p_value=tail, joined=len(joined))
        if joined:
            Dmed = np.median([r["D"] for r, _ in joined])
            comp = [(r, q) for r, q in joined if abs(r["D"] - Dmed) <= 0.25 * Dmed]
            tab = {}
            for near in (True, False):
                grp = [q for r, q in comp if r["near"] == near]
                stuck = sum(q["tclear"] < 0 for q in grp)
                wrong = sum(q["tclear"] >= 0 and q["w_at_clear"] for q in grp)
                tab[near] = (len(grp), stuck, wrong)
            (n1, s1, w1), (n0, s0, w0) = tab[True], tab[False]
            pf_stuck = fisher_greater(s1, n1 - s1, s0, n0 - s0) if n1 and n0 else float("nan")
            pf_bad = fisher_greater(s1 + w1, n1 - s1 - w1, s0 + w0, n0 - s0 - w0) if n1 and n0 else float("nan")
            out += [f"joined with replay: {len(joined)}; comparable distance (D within 25% of the median {Dmed:.0f}): {len(comp)}",
                    "| orientation | pairs | never cleared in +40L | cleared the wrong way (logical) |", "|---|---|---|---|",
                    f"| near-axis | {n1} | {s1} | {w1} |", f"| off-axis | {n0} | {s0} | {w0} |",
                    f"one-sided Fisher: near-axis more often stuck p = {pf_stuck:.3g}; stuck or wrong way p = {pf_bad:.3g}"]
            res[key].update(table=dict(near=tab[True], off=tab[False]), p_stuck=pf_stuck, p_bad=pf_bad)
        out.append("")
    all_two = [r for rows in rows_by_key.values() for r in rows if r["ndef"] == 2]
    if all_two:
        x = sum(r["near"] for r in all_two)
        tail, mean = poisson_binomial_tail([r["p0"] for r in all_two], x)
        out.append(f"POOLED: near-axis {x} of {len(all_two)} vs {mean:.1f} expected; P(X >= {x}) = {tail:.2e}; "
                   f"P3 pass rule (p < 0.01): {'MET' if tail < 0.01 else 'NOT MET'}")
        res["pooled"] = dict(near=x, n=len(all_two), expected=mean, p_value=tail)
    return "\n".join(out) + "\n", res


def _count(it):
    c = {}
    for v in it:
        c[v] = c.get(v, 0) + 1
    return c


def cmd_run(a):
    d = Path(a.out) / "diag1"
    rows_by_key, caps = {}, []
    for f in sorted(d.glob("states-score-v10-big-L*-p*-s*.pt")):
        S = torch.load(f)
        if len(S["states"]) >= 50:
            caps.append(f.name)
        for s in S["states"]:
            s["seed"], s["p"] = S["seed"], S["p"]
        rows_by_key.setdefault((S["L"], round(S["p"], 4)), []).extend(analyse(S["states"], S["L"]))
    replay_by_key = {}
    for f in sorted(d.glob("replay-v10-big-L*-p*.json")):
        r = json.loads(f.read_text())
        replay_by_key[(r["L"], round(r["p"], 4))] = {(x["seed"], x["shot"]): x for x in r["rows"]
                                                     if x["variant"] == "original"}
    text, res = summarise(rows_by_key, replay_by_key)
    text += f"\nfiles at the 50-state cap: {caps if caps else 'none'}\ncode {CODE_HASH}\n"
    (d / "p3-axis.md").write_text(text)
    (d / "p3-axis.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1, default=str))
    print(text)


def cmd_check(a):
    ok = True

    def report(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)

    f1, t1 = null_fraction(111, 48)
    f2, t2 = null_fraction(155, 65)
    report("1 null fractions = reviewer's counts (36/192 at L111 D48; 52/260 at L155 D65)",
           abs(f1 - 36 / 192) < 1e-12 and t1 == 192 and abs(f2 - 52 / 260) < 1e-12 and t2 == 260, f"{f1:.4f} {f2:.4f}")
    f3, t3 = null_fraction(11, 9)
    brute = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if abs(dx) + abs(dy) == 9]
    report("1b null near the torus size (L11, D9) = brute-force enumeration", t3 == len(brute) and
           abs(f3 - sum(min(abs(x), abs(y)) / 9 < NEAR for x, y in brute) / len(brute)) < 1e-12, f"{t3} offsets")
    report("2 minimum image across the boundary", min_image((1, 2), (109, 5), 111) == [-3, 3] and
           min_image((0, 0), (55, 56), 111) == [55, -55])
    ps = [0.2] * 30
    tail, mean = poisson_binomial_tail(ps, 9)
    exact = sum(comb(30, i) * 0.2 ** i * 0.8 ** (30 - i) for i in range(9, 31))
    report("3 Poisson-binomial = binomial when all p equal", abs(tail - exact) < 1e-12 and abs(mean - 6) < 1e-9)
    report("3b Fisher one-sided on a known table", abs(fisher_greater(3, 1, 1, 3) - 17 / 70) < 1e-12)
    # 4. synthetic states: drawn from the null -> no signal; all on an axis -> strong signal (positive control)
    rng = np.random.default_rng(1)
    L = 111

    def state(dx, dy):
        e = torch.zeros(2, L, L, dtype=torch.bool)
        i, j = 30, 30
        for _ in range(abs(dx)):
            e[0, i % L, j % L] ^= True; i += 1 if dx > 0 else 0
            if dx < 0:
                e[0, (i - 1) % L, j % L] ^= True; e[0, i % L, j % L] ^= True; i -= 1
        for _ in range(abs(dy)):
            e[1, i % L, j % L] ^= True; j += 1 if dy > 0 else 0
            if dy < 0:
                e[1, i % L, (j - 1) % L] ^= True; e[1, i % L, j % L] ^= True; j -= 1
        return dict(e=e, corr=torch.zeros_like(e), shot=0)
    offs = [(dx, dy) for dx in range(-55, 56) for dy in range(-55, 56) if abs(dx) + abs(dy) == 48]
    null_states = [state(*offs[i]) for i in rng.integers(len(offs), size=200)]
    rows = analyse(null_states, L)
    okd = all(r["ndef"] == 2 and r["D"] == 48 for r in rows)
    x = sum(r["near"] for r in rows)
    tail, mean = poisson_binomial_tail([r["p0"] for r in rows], x)
    report("4 synthetic pairs: two defects at the intended distance (both signs, both axes)", okd)
    report("4b negative control: pairs drawn from the null give no significant excess", tail > 0.01, f"p = {tail:.2f}")
    axis = [state(48, 0), state(0, -48), state(-47, 1), state(2, 46)] * 10
    rows = analyse(axis, L)
    x = sum(r["near"] for r in rows)
    tail, _ = poisson_binomial_tail([r["p0"] for r in rows], x)
    report("4c positive control: near-axis pairs give a tiny p-value", tail < 1e-10, f"p = {tail:.1e}")
    print("P3 CHECK OK" if ok else "P3 CHECK FAILED")
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    sub.add_parser("check")
    a = ap.parse_args()
    {"run": cmd_run, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
