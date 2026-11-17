"""Two cheap tests of trained models (no training).

quant : every number a site stores (its memory) and every number it sends (its message) is rounded to b bits at every
        message tick. b bits = 2^b - 1 evenly spaced levels in [-1, 1], zero included (all these numbers come out of a
        tanh, so they already lie in [-1, 1]). Same noise for every b, so the comparison is paired.
        Bits per site: C * b stored; M * b sent per tick (the same message to all four neighbours).
online: long runs with no clean-up phase, as in Paletta (arXiv:2603.02328): noise on data and readings every round for up
        to 16 L rounds. At checkpoints, a copy of the leftover error (true error xor corrections) is fixed by an ideal
        noiseless matching; a logical flip remains -> failure at that time. With two logical qubits the flip probability
        saturates at 3/4, so the logical error rate per round is  lam(tau) = 1 - (1 - 4/3 P(tau))^(1/tau).
        A rule that is stable over long times gives lam(tau) roughly constant in tau.

  python decoder_tests.py check
  python decoder_tests.py quant  --labels im-same,v7-wide --bits 0,8,6,4,3 --points 19:0.018,27:0.018,39:0.015 --shots 10000
  python decoder_tests.py online --labels im-same,v7-wide --Ls 13,19,27 --ps 0.012,0.015,0.018 --shots 4000
  python decoder_tests.py summary --out tests-summary.md
(bits 0 = no rounding)
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from decoder_v41 import build, evaluate, pick_device, rnd, shift_to, synd, winding


def load(label, dev, model_file="model.pt"):
    src = Path("runs") / f"nca-{label}"
    cfg = json.loads((src / "config.json").read_text())
    model = build(cfg).to(dev)
    model.load_state_dict(torch.load(src / model_file, map_location=dev))
    model.eval()
    return cfg, model


# ------------------------------------------------------------------ quantised round

def quantise(v, b):
    n = 2 ** (b - 1) - 1
    return torch.round(v.clamp(-1, 1) * n) / n


def quant_round(model, b, ste=False):
    """a copy of Rule.round (plain update, same message to all neighbours, link heads) that rounds memory and
    messages to b bits after every tick; b = 0: no rounding (must reproduce Rule.round exactly).
    ste=True (for training): the forward pass is rounded, the gradient passes through as if it were not
    ("straight-through" estimate), so the weights can learn to live with the rounding."""
    if model.gru or model.dirmsg or model.action != "link":
        raise SystemExit("quant supports the plain link rule only")
    if not b:
        q = lambda v: v
    elif ste:
        q = lambda v: v + (quantise(v, b) - v).detach()
    else:
        q = lambda v: quantise(v, b)

    def round_q(h, x):
        for _ in range(model.K):
            m = q(torch.tanh(model.msg(h)))
            inc = [shift_to(m, k) for k in range(4)]
            h = q(torch.tanh(h + model.upd(torch.cat([h, *inc, x], 1))))
        l0 = model.act[0](torch.cat([h, shift_to(h, 0)], 1))[:, 0]
        l1 = model.act[1](torch.cat([h, shift_to(h, 1)], 1))[:, 0]
        return h, torch.stack([l0, l1], 1)
    return round_q


def batch_for(L):
    return 512 if L < 27 else (256 if L < 39 else 128)


def cmd_quant(a):
    dev = pick_device(a.device)
    pts = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in a.points.split(",")]
    for label in a.labels.split(","):
        cfg, model = load(label, dev)
        out = Path("runs") / f"nca-{label}" / "test-quant.jsonl"
        for b in map(int, a.bits.split(",")):
            m2 = copy.copy(model)
            m2.round = quant_round(model, b)
            for i, (L, p) in enumerate(pts):
                t0 = time.monotonic()
                r = evaluate(m2, L, p, a.shots, batch_for(L), seed=a.seed + 100 * L + i, dev=dev)
                rec = dict(label=label, C=model.C, M=model.M, K=model.K, bits=b, L=L, p=p, shots=r["n"],
                           strict=r["strict"], leftover_runs=r["residual"],
                           bits_stored=model.C * (b or 32), bits_sent_per_tick=model.M * (b or 32))
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{label} bits {b or 'none':>4} L={L} p={100*p:.1f}%: {r['strict']}/{r['n']} "
                      f"({r['strict']/r['n']:.4f})  {time.monotonic()-t0:.0f}s", flush=True)


def cmd_qeval(a):
    """decoder_v41.py eval with the rounding switched on: same noise (seed per point, batch 250), same eval-<tag>.jsonl
    format, so results are paired with earlier evals. Bits from --bits, else from the model's config (qbits)."""
    from decoder_v41 import LAKE_REF, wilson
    dev = pick_device(a.device)
    cfg, model = load(a.label, dev, a.model)
    b = a.bits if a.bits >= 0 else int(cfg.get("qbits", 0))
    model.round = quant_round(model, b)
    out = Path("runs") / f"nca-{a.label}"
    jl = out / f"eval-{a.tag}.jsonl"
    rows = [json.loads(x) for x in jl.read_text().splitlines()] if jl.exists() else []
    done = {(r["L"], r["p"]) for r in rows}
    with torch.no_grad():
        for L in map(int, a.Ls.split(",")):
            for p in map(float, a.ps.split(",")):
                if (L, p) in done:
                    continue
                t0 = time.monotonic()
                r = evaluate(model, L, p, a.shots, 250, seed=(L * 1000003 + int(p * 1e6) + a.seed) % 2**31, dev=dev)
                n = r["n"]
                row = dict(L=L, p=p, shots=n, rate_strict=r["strict"] / n, ci_strict=wilson(r["strict"], n),
                           rate_native=r["native"] / n, residual=r["residual"], leftover=r["leftover"],
                           stuck=r["stuck"], lake=LAKE_REF.get(L, {}).get(p), seconds=time.monotonic() - t0,
                           qbits=b, model=a.model)
                rows.append(row)
                with jl.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{a.label} [{a.model}, {b or 'no'} bits] L {L:3d} p {p:.4f}: strict {row['rate_strict']:.4f} "
                      f"leftover runs {r['residual']}  ({row['seconds']:.0f}s)", flush=True)


# ------------------------------------------------------------------ long online runs

_SP = {}


def ideal_fix(r):
    """minimum-weight matching of the leftover error's true defects -> correcting link set (B, 2, L, L) bool."""
    B, _, L, _ = r.shape
    if L not in _SP:
        import pymatching
        import scipy.sparse as sp
        N = L * L
        ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        s = (ii * L + jj).ravel()
        sx = (((ii + 1) % L) * L + jj).ravel()
        sy = (ii * L + (jj + 1) % L).ravel()
        rows = np.concatenate([s, sx, s, sy])
        cols = np.concatenate([s, s, N + s, N + s])
        H = sp.csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(N, 2 * N))
        F = sp.identity(2 * N, dtype=np.uint8, format="csc")
        _SP[L] = pymatching.Matching.from_check_matrix(H, faults_matrix=F)
    syn = synd(r).flatten(1).cpu().numpy().astype(np.uint8)
    fix = np.zeros((B, 2 * L * L), dtype=np.uint8)
    hit = syn.any(1)
    if hit.any():
        fix[hit] = _SP[L].decode_batch(syn[hit])
    return torch.from_numpy(fix.astype(bool)).view(B, 2, L, L).to(r.device)


def online(model, L, p, shots, batch, seed, dev, checkpoints):
    gen = torch.Generator().manual_seed(seed)
    fails = {c: 0 for c in checkpoints}
    n = 0
    T = max(checkpoints)
    with torch.no_grad():
        while n < shots:
            B = min(batch, shots - n)
            e = torch.zeros(B, 2, L, L, dtype=torch.bool, device=dev)
            corr = torch.zeros_like(e)
            prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
            h = model.init_state(B, L, dev)
            for t in range(1, T + 1):
                e = e ^ (rnd(e.shape, gen, dev) < p)
                td = synd(e ^ corr)
                m = td ^ (rnd(td.shape, gen, dev) < p)
                h, logits = model.round(h, torch.stack([m, m ^ prev], 1).float())
                prev = m
                corr = corr ^ model.flips(model.greedy(logits))
                if t in fails:
                    r = e ^ corr
                    fails[t] += int(winding(r ^ ideal_fix(r)).sum())
            n += B
    return fails, n


def rate(P, tau):
    x = 1 - 4.0 / 3.0 * P
    return float("nan") if x <= 0 else 1 - x ** (1.0 / tau)


def cmd_online(a):
    dev = pick_device(a.device)
    for label in a.labels.split(","):
        cfg, model = load(label, dev)
        out = Path("runs") / f"nca-{label}" / "test-online.jsonl"
        for L in map(int, a.Ls.split(",")):
            cps = [k * L for k in (1, 2, 4, 8, 16)]
            for p in map(float, a.ps.split(",")):
                t0 = time.monotonic()
                fails, n = online(model, L, p, a.shots, 1024 if L < 27 else 512, a.seed + 1000 * L + int(1e4 * p),
                                  dev, cps)
                for c in cps:
                    P = fails[c] / n
                    rec = dict(label=label, L=L, p=p, tau=c, tau_over_L=c // L, shots=n, flips=fails[c], P=P,
                               rate_per_round=rate(P, c))
                    with out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                print(f"{label} L={L} p={100*p:.1f}%: " + "  ".join(
                    f"{c//L}L {fails[c]}/{n} (rate {rate(fails[c]/n, c):.2e})" for c in cps)
                      + f"  {time.monotonic()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ summary

def cmd_summary(a):
    lines = ["# Tests of trained models (SIMULATED)", ""]
    labels = a.labels.split(",")
    q = [json.loads(x) for n in labels for f in [Path(f"runs/nca-{n}/test-quant.jsonl")] if f.exists()
         for x in f.read_text().splitlines()]
    if q:
        lines += ["## Memory and messages rounded to b bits (failures / runs; same noise for every b)", "",
                  "| model | bits | stored bits/site | sent bits/tick | " +
                  " | ".join(f"L={L} p={100*p:.1f}%" for L, p in sorted({(r['L'], r['p']) for r in q})) + " |",
                  "|---|---|---|---|" + "---|" * len({(r['L'], r['p']) for r in q})]
        for n in labels:
            for b in sorted({r["bits"] for r in q if r["label"] == n}, key=lambda b: -(b or 99)):
                rs = {(r["L"], r["p"]): r for r in q if r["label"] == n and r["bits"] == b}
                any_r = next(iter(rs.values()))
                cells = [f"{rs[k]['strict']}/{rs[k]['shots']}" if k in rs else "-"
                         for k in sorted({(r['L'], r['p']) for r in q})]
                lines.append(f"| {n} | {b or 'none (32-bit floats)'} | {any_r['bits_stored'] or '-'} | "
                             f"{any_r['bits_sent_per_tick'] or '-'} | " + " | ".join(cells) + " |")
        lines.append("")
    o = [json.loads(x) for n in labels for f in [Path(f"runs/nca-{n}/test-online.jsonl")] if f.exists()
         for x in f.read_text().splitlines()]
    if o:
        lines += ["## Long online runs: logical error rate per round at tau = L ... 16 L (flips / runs in brackets)", "",
                  "| model | L | p | L | 2L | 4L | 8L | 16L |", "|---|---|---|---|---|---|---|---|"]
        for n in labels:
            for L, p in sorted({(r["L"], r["p"]) for r in o if r["label"] == n}):
                rs = {r["tau_over_L"]: r for r in o if r["label"] == n and r["L"] == L and r["p"] == p}
                cells = [f"{rs[k]['rate_per_round']:.2e} ({rs[k]['flips']}/{rs[k]['shots']})" if k in rs else "-"
                         for k in (1, 2, 4, 8, 16)]
                lines.append(f"| {n} | {L} | {100*p:.1f}% | " + " | ".join(cells) + " |")
        lines.append("")
    Path(a.out).write_text("\n".join(lines))
    print("\n".join(lines))


# ------------------------------------------------------------------ self-tests

def cmd_check(a):
    ok_all = True
    torch.manual_seed(0)
    model = build(dict(C=16, M=8, K=3, hidden=64, gru=False, action="link", dirmsg=False))
    for prm in model.parameters():
        prm.data.normal_(0, 0.3)
    m64 = copy.deepcopy(model).double()
    L = 9
    x = (torch.rand(4, 2, L, L) < 0.1).double()
    with torch.no_grad():
        rq = quant_round(m64, 0)
        h1, l1 = m64.round(m64.init_state(4, L, "cpu").double(), x)
        h2, l2 = rq(m64.init_state(4, L, "cpu").double(), x)
        for _ in range(3):
            h1, l1 = m64.round(h1, x); h2, l2 = rq(h2, x)
    err = float((l1 - l2).abs().max())
    ok = err < 1e-9; ok_all &= ok
    print(f"1. unrounded copy of the round reproduces the model (double precision): max diff {err:.1e}",
          "OK" if ok else "FAIL")
    v = torch.linspace(-1.2, 1.2, 10001)
    for b in (8, 4, 3, 2):
        k = len(torch.unique(quantise(v, b)))
        ok = k == 2 ** b - 1 and bool((quantise(torch.zeros(1), b) == 0).all()); ok_all &= ok
        print(f"2. {b} bits -> {k} levels (expected {2**b-1}), zero kept:", "OK" if ok else "FAIL")
    gen = torch.Generator().manual_seed(2)
    r = torch.rand(64, 2, 13, 13, generator=gen) < 0.03
    fix = ideal_fix(r)
    ok = bool((synd(fix) == synd(r)).all()); ok_all &= ok
    print("3. ideal matching removes all defects:", "OK" if ok else "FAIL")
    r = torch.zeros(8, 2, 13, 13, dtype=torch.bool); r[:, 0, 3, 4] = True; r[:, 1, 7, 2] = True
    ok = not bool(winding(r ^ ideal_fix(r)).any()); ok_all &= ok
    print("4. two isolated wrong links are fixed without a logical flip:", "OK" if ok else "FAIL")
    idle = copy.deepcopy(model)                       # a rule that never flips anything
    with torch.no_grad():
        for hd in idle.act:
            hd[-1].bias.fill_(-1e3)
    fails, n = online(idle, 7, 0.0, 16, 16, 0, torch.device("cpu"), [7, 14])
    ok5 = fails == {7: 0, 14: 0}
    fails, n = online(idle, 7, 0.01, 400, 400, 0, torch.device("cpu"), [7, 14])
    ok5 &= fails[7] < fails[14] and fails[14] > 0                    # with noise and no decoding, flips build up over time
    ok_all &= ok5
    print(f"5. idle rule: no noise -> no flips; noise -> flips grow with time ({fails[7]} then {fails[14]} of 400):",
          "OK" if ok5 else "FAIL")
    ok = abs(rate(0.0, 10)) < 1e-12 and abs(rate(1 - (0.99 ** 10), 10) - (1 - (1 - 4 / 3 * (1 - 0.99 ** 10)) ** 0.1)) < 1e-12
    ok_all &= ok
    print("6. rate formula:", "OK" if ok else "FAIL")
    if not ok_all:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    q = sub.add_parser("quant")
    q.add_argument("--labels", required=True); q.add_argument("--bits", default="0,8,6,4,3")
    q.add_argument("--points", default="19:0.018,27:0.018,39:0.015"); q.add_argument("--shots", type=int, default=10000)
    q.add_argument("--seed", type=int, default=31); q.add_argument("--device", default="auto")
    o = sub.add_parser("online")
    o.add_argument("--labels", required=True); o.add_argument("--Ls", default="13,19,27")
    o.add_argument("--ps", default="0.012,0.015,0.018"); o.add_argument("--shots", type=int, default=4000)
    o.add_argument("--seed", type=int, default=41); o.add_argument("--device", default="auto")
    e = sub.add_parser("qeval")
    e.add_argument("--label", required=True); e.add_argument("--model", default="model.pt")
    e.add_argument("--bits", type=int, default=-1, help="-1: take qbits from the model's config")
    e.add_argument("--Ls", required=True); e.add_argument("--ps", required=True)
    e.add_argument("--shots", type=int, default=10000); e.add_argument("--seed", type=int, default=7)
    e.add_argument("--tag", required=True); e.add_argument("--device", default="auto")
    s = sub.add_parser("summary")
    s.add_argument("--labels", default="im-same,v7-wide"); s.add_argument("--out", default="tests-summary.md")
    a = ap.parse_args()
    {"check": cmd_check, "quant": cmd_quant, "online": cmd_online, "summary": cmd_summary,
     "qeval": cmd_qeval}[a.cmd](a)


if __name__ == "__main__":
    main()
