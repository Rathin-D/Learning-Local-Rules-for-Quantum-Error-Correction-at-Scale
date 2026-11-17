"""diag_largeL2.py = diag_largeL.py + options --sub (results sub-folder) and --cks (checkpoints) for score, --sub/--name/
--noisy-orients for pairs. A separate file so that diag1 jobs still running on SubMIT keep using the unchanged diag_largeL.py.

Large-L diagnostics after the w1b failure (report/failure-largeL-20261114.md, referee report review-largeL-20261114.md).
All results SIMULATED. Nothing here trains or changes a model.

Commands
  score   FULL evolution (every run continues to 20L, no shortcut), per-run records, for one or two models on the SAME
          noise (paired). Scored at c = 0, 2, 6, 10, 20 noiseless L-blocks:
            strict  = defects left OR logical error (winding); left = defects left;
            matched = logical error after global MWPM on the true leftover syndrome (local evolution + global readout);
            lake    = Lake-style score (his released code): stop at the first clean noiseless round (up to 2L) and score
                      the winding then; a run not clean after 2L is scored by its winding anyway (NOT counted as failed).
          Also: what the old evaluate_long shortcut would have reported (runs clean at 6L keep their 6L result), and per-run
          disagreements with full evolution; runs that re-open after being clean; runs whose logical class changes while
          clean ("silent"). Optionally saves the full state (e, corr, prev, lastflip, memory h) of runs unfinished at 6L.
  replay  continues saved unfinished states with (a) their own memory, (b) zero memory, (c) memory warmed on a clean
          lattice; records time to clear, defect count and matching weight over time, final strict/matched.
  pairs   one planted defect pair at separations 2..48 (straight and staircase), no noise, zero or warmed memory: time to
          clear, logical error, separation over time. Optional: the same pair under background noise vs a paired control
          (same noise, no pair).
  summary merges score/replay/pairs jsons into markdown.
  check   self-checks with known answers and deliberately broken cases (CPU, ~1 min).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decoder_v41 import Rule, rnd, synd, winding  # noqa: E402
from decoder_v7 import observe  # noqa: E402
from decoder_tests import ideal_fix  # noqa: E402

DIAG_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
DEP_HASH = hashlib.sha256(b"".join((HERE / f).read_bytes() for f in
                                   ("decoder_v41.py", "decoder_v7.py", "decoder_tests.py"))).hexdigest()[:12]
CKS = (0, 2, 6, 10, 20)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * (ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [max(0.0, c - h), min(1.0, c + h)]


def tdist(a, b, L):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return min(dx, L - dx) + min(dy, L - dy)


# ------------------------------------------------------------------------------------------------ score (full evolution)

def run_full(model, L, p, shots, batch, seed, dev, cks=CKS, save_max=0, save_at=6):
    """returns per-run numpy arrays and saved states. Same loop and same random numbers as decoder_v7.evaluate_multi."""
    gen = torch.Generator().manual_seed(seed)
    xin = getattr(model, "xin", 2)
    cks = tuple(sorted(cks))
    T = L + max(cks) * L
    marks = {L + c * L: c for c in cks}
    rec = {f"{k}_{c}": [] for c in cks for k in ("left", "w", "matched")}
    for k in ("lake", "lake_unfinished", "first_clean", "reopen", "silent", "ndef_save"):
        rec[k] = []
    noise_hashes, saved = [], []
    n = 0
    with torch.no_grad():
        while n < shots:
            B = min(batch, shots - n)
            e = torch.zeros(B, 2, L, L, dtype=torch.bool, device=dev)
            corr = torch.zeros_like(e)
            prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
            lastflip = torch.zeros_like(e)
            h = model.init_state(B, L, dev)
            first = torch.full((B,), -1, dtype=torch.long, device=dev)
            wfirst = torch.zeros(B, dtype=torch.bool, device=dev)
            lake = torch.zeros(B, dtype=torch.bool, device=dev)
            lake_done = torch.zeros(B, dtype=torch.bool, device=dev)
            lake_unf = torch.zeros(B, dtype=torch.bool, device=dev)
            reopen = torch.zeros(B, dtype=torch.bool, device=dev)
            ndef_save = torch.full((B,), -1, dtype=torch.long, device=dev)
            for t in range(T):
                noisy = t < L
                if noisy:
                    e = e ^ (rnd(e.shape, gen, dev) < p)
                td = synd(e ^ corr)
                m = td ^ (rnd(td.shape, gen, dev) < p) if noisy else td
                h, logits = model.round(h, observe(m, prev, lastflip, xin))
                flips = model.flips(model.greedy(logits))
                prev, lastflip = m, flips
                corr = corr ^ flips
                tt = t + 1
                if tt == L:
                    noise_hashes.append(hashlib.sha256(e.cpu().numpy().tobytes()).hexdigest()[:16])
                if tt < L:
                    continue
                k = tt - L                                         # noiseless rounds done so far
                d = e ^ corr
                sy = synd(d)
                left = sy.flatten(1).any(1)
                w = winding(d)
                newc = (~left) & (first < 0)
                wfirst = torch.where(newc, w, wfirst)
                first = torch.where(newc, torch.full_like(first, k), first)
                reopen = reopen | ((first >= 0) & left)
                if k <= 2 * L:
                    take = newc & ~lake_done
                    lake = torch.where(take, w, lake)
                    lake_done = lake_done | take
                    if k == 2 * L:
                        rest = ~lake_done
                        lake = torch.where(rest, w, lake)
                        lake_unf = rest.clone()
                        lake_done = torch.ones_like(lake_done)
                if tt in marks:
                    c = marks[tt]
                    rec[f"left_{c}"].append(left.cpu().numpy())
                    rec[f"w_{c}"].append(w.cpu().numpy())
                    rec[f"matched_{c}"].append(winding(d ^ ideal_fix(d)).cpu().numpy())
                if save_max and tt == L + save_at * L and save_at <= max(cks):
                    ndef_save = sy.flatten(1).sum(1)
                    for b in torch.nonzero(left).flatten().tolist():
                        if len(saved) < save_max:
                            saved.append(dict(shot=n + b, e=e[b].cpu(), corr=corr[b].cpu(), prev=prev[b].cpu(),
                                              lastflip=lastflip[b].cpu(), h=h[b].cpu()))
                if tt == T:
                    rec["silent"].append(((first >= 0) & ~left & (w != wfirst)).cpu().numpy())
            rec["lake"].append(lake.cpu().numpy())
            rec["lake_unfinished"].append(lake_unf.cpu().numpy())
            rec["first_clean"].append(first.cpu().numpy())
            rec["reopen"].append(reopen.cpu().numpy())
            rec["ndef_save"].append(ndef_save.cpu().numpy())
            n += B
    arr = {k: np.concatenate(v) for k, v in rec.items() if v}
    return n, arr, noise_hashes, saved


def counts_from(arr, cks=CKS):
    out = {}
    n = len(arr["lake"])
    for c in cks:
        strict = arr[f"left_{c}"] | arr[f"w_{c}"]
        out[f"strict_{c}L"] = int(strict.sum())
        out[f"left_{c}L"] = int(arr[f"left_{c}"].sum())
        out[f"matched_{c}L"] = int(arr[f"matched_{c}"].sum())
    out["lake"] = int(arr["lake"].sum())
    out["lake_unfinished"] = int(arr["lake_unfinished"].sum())
    out["reopen"] = int(arr["reopen"].sum())
    out["silent"] = int(arr["silent"].sum())
    out["never_clean"] = int((arr["first_clean"] < 0).sum())
    if 6 in cks:                                                  # what evaluate_long's shortcut would have said
        clean6 = ~arr["left_6"]
        for c in cks:
            if c > 6:
                full = arr[f"left_{c}"] | arr[f"w_{c}"]
                short = np.where(clean6, arr["w_6"], full)
                out[f"shortcut_strict_{c}L"] = int(short.sum())
                out[f"shortcut_disagree_{c}L"] = int((short != full).sum())
    out["n"] = n
    return out


def cmd_score(a):
    os.chdir(HERE)
    from decoder_v7 import file_hash, load_model
    dev = torch.device(a.device)
    out = Path(a.out) / a.sub
    out.mkdir(parents=True, exist_ok=True)
    cks = tuple(sorted(int(c) for c in a.cks.split(",")))
    if 2 not in cks:
        raise SystemExit("--cks must contain 2 (the Lake-style score needs the 2L clean-up)")
    name = f"score-{a.label}-L{a.L}-p{a.p:.4f}-s{a.seed}"
    jf = out / f"{name}.json"
    if jf.exists():
        print(f"{jf} exists; skipping", flush=True)
        return
    models = a.models.split(",")
    res = dict(cmd="score", label=a.label, L=a.L, p=a.p, shots=a.shots, batch=a.batch, seed=a.seed, cks=list(cks),
               diag_hash=DIAG_HASH, dep_hash=DEP_HASH, device=str(dev), host=os.uname().nodename,
               job=os.environ.get("SLURM_JOB_ID"), torch=torch.__version__, tf32=torch.backends.cuda.matmul.allow_tf32,
               models={})
    per_run = {}
    hashes = {}
    for i, mf in enumerate(models):
        cfg, m = load_model(a.label, mf, dev)
        m.eval()
        t0 = time.monotonic()
        n, arr, nh, saved = run_full(m, a.L, a.p, a.shots, a.batch, a.seed, dev, cks=cks,
                                     save_max=a.save if i == 0 else 0)
        c = counts_from(arr, cks)
        c.update(model_hash=file_hash(Path("runs") / f"nca-{a.label}" / mf), seconds=round(time.monotonic() - t0, 1))
        res["models"][mf] = c
        hashes[mf] = nh
        for k, v in arr.items():
            per_run[f"{mf}|{k}"] = v
        if saved:
            torch.save(dict(L=a.L, p=a.p, label=a.label, model=mf, seed=a.seed, states=saved,
                            diag_hash=DIAG_HASH), out / f"states-{name}.pt")
        print(f"{name} [{mf}] {n} runs ({c['seconds']:.0f}s): " + "  ".join(
            f"{k} {v}" for k, v in c.items() if k not in ("model_hash", "seconds")) +
            (f"  saved {len(saved)} states" if saved else ""), flush=True)
    if len(models) == 2:
        A, B = models
        res["paired_noise_identical"] = hashes[A] == hashes[B]
        pair = {}
        for c in cks:
            sa = per_run[f"{A}|left_{c}"] | per_run[f"{A}|w_{c}"]
            sb = per_run[f"{B}|left_{c}"] | per_run[f"{B}|w_{c}"]
            pair[f"strict_{c}L"] = dict(both=int((sa & sb).sum()), only_A=int((sa & ~sb).sum()),
                                        only_B=int((~sa & sb).sum()))
            ma, mb = per_run[f"{A}|matched_{c}"], per_run[f"{B}|matched_{c}"]
            pair[f"matched_{c}L"] = dict(both=int((ma & mb).sum()), only_A=int((ma & ~mb).sum()),
                                         only_B=int((~ma & mb).sum()))
        res["paired"] = dict(A=A, B=B, **pair)
        print(f"paired noise identical: {res['paired_noise_identical']}; strict {max(cks)}L both/onlyA/onlyB "
              f"{pair[f'strict_{max(cks)}L']}", flush=True)
    np.savez_compressed(out / f"{name}.npz", **{k.replace("|", "__"): v for k, v in per_run.items()})
    tmp = jf.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, jf)


# ------------------------------------------------------------------------------------------------ replay

def warm_state(model, L, dev, rounds):
    """memory after `rounds` rounds on a lattice with no defects at all; also returns how many links it flipped."""
    xin = getattr(model, "xin", 2)
    h = model.init_state(1, L, dev)
    z = torch.zeros(1, L, L, dtype=torch.bool, device=dev)
    lf = torch.zeros(1, 2, L, L, dtype=torch.bool, device=dev)
    nflip = 0
    with torch.no_grad():
        for _ in range(rounds):
            h, logits = model.round(h, observe(z, z, lf, xin))
            nflip += int(model.flips(model.greedy(logits)).sum())
    return h, nflip


def geometry(d, L):
    """defect count, nearest-neighbour distances (torus), and total MWPM weight of the leftover syndrome."""
    sy = synd(d)
    out = []
    fixw = ideal_fix(d).flatten(1).sum(1).cpu().numpy()
    for b in range(d.shape[0]):
        pts = torch.nonzero(sy[b]).cpu().tolist()
        if len(pts) >= 2:
            nn = [min(tdist(pts[i], pts[j], L) for j in range(len(pts)) if j != i) for i in range(len(pts))]
            out.append(dict(ndef=len(pts), nn_min=int(min(nn)), nn_med=float(np.median(nn)), nn_max=int(max(nn)),
                            mwpm_weight=int(fixw[b])))
        else:
            out.append(dict(ndef=len(pts), nn_min=-1, nn_med=-1, nn_max=-1, mwpm_weight=int(fixw[b])))
    return out


def replay_states(model, states, L, dev, extra, variants, batch, warm_rounds, marks_extra, return_final=False):
    """continue saved states (all from the same L) under each memory variant for `extra` noiseless rounds.
    'original' keeps h, prev, lastflip exactly (an exact continuation of the saved run); 'zero' and 'warm' replace the
    memory and set prev = current reading, lastflip = none (the rule sees no 'change' at the restart)."""
    xin = getattr(model, "xin", 2)
    hw, warm_flips = warm_state(model, L, dev, warm_rounds) if "warm" in variants else (None, 0)
    rows, finals = [], []
    with torch.no_grad():
        for v in variants:
            for s0 in range(0, len(states), batch):
                S = states[s0:s0 + batch]
                e = torch.stack([s["e"] for s in S]).to(dev)
                corr = torch.stack([s["corr"] for s in S]).to(dev)
                B = e.shape[0]
                if v == "original":
                    h = torch.stack([s["h"] for s in S]).to(dev)
                    prev = torch.stack([s["prev"] for s in S]).to(dev)
                    lastflip = torch.stack([s["lastflip"] for s in S]).to(dev)
                else:
                    h = model.init_state(B, L, dev) if v == "zero" else hw.expand(B, -1, -1, -1).clone()
                    prev = synd(e ^ corr)
                    lastflip = torch.zeros_like(e)
                if v != "original":
                    assert (v == "zero" and float(h.abs().sum()) == 0.0) or v == "warm"
                geo0 = geometry(e ^ corr, L)
                tclear = torch.full((B,), -1, dtype=torch.long, device=dev)
                wclear = torch.zeros(B, dtype=torch.bool, device=dev)
                reopen = torch.zeros(B, dtype=torch.bool, device=dev)
                track = {}
                for t in range(extra):
                    td = synd(e ^ corr)
                    h, logits = model.round(h, observe(td, prev, lastflip, xin))
                    flips = model.flips(model.greedy(logits))
                    prev, lastflip = td, flips
                    corr = corr ^ flips
                    d = e ^ corr
                    left = synd(d).flatten(1).any(1)
                    w = winding(d)
                    newc = (~left) & (tclear < 0)
                    wclear = torch.where(newc, w, wclear)
                    tclear = torch.where(newc, torch.full_like(tclear, t + 1), tclear)
                    reopen = reopen | ((tclear >= 0) & left)
                    if t + 1 in marks_extra:
                        g = geometry(d, L)
                        track[t + 1] = dict(left=left.cpu().numpy().tolist(), w=w.cpu().numpy().tolist(),
                                            ndef=[x["ndef"] for x in g], mwpm_weight=[x["mwpm_weight"] for x in g],
                                            matched=winding(d ^ ideal_fix(d)).cpu().numpy().tolist())
                if return_final:
                    finals += [corr[b].cpu() for b in range(B)]
                for b in range(B):
                    rows.append(dict(variant=v, shot=S[b]["shot"], seed=S[b].get("seed"), p=S[b].get("p"),
                                     start=geo0[b], tclear=int(tclear[b]), w_at_clear=bool(wclear[b]),
                                     reopen=bool(reopen[b]),
                                     track={str(k): {kk: vv[b] for kk, vv in tr.items()} for k, tr in track.items()}))
    if return_final:
        return rows, warm_flips, finals
    return rows, warm_flips


def cmd_replay(a):
    os.chdir(HERE)
    from decoder_v7 import file_hash, load_model
    dev = torch.device(a.device)
    out = Path(a.out) / a.sub
    files = sorted(out.glob(f"states-score-{a.label}-L{a.L}-p{a.p:.4f}-*.pt"))
    if not files:
        raise SystemExit(f"no saved states for L={a.L}, p={a.p} in {out}")
    states, models = [], set()
    for f in files:
        d = torch.load(f)
        models.add(d["model"])
        for s in d["states"]:
            s["seed"], s["p"] = d["seed"], d["p"]
        states += d["states"]
    if len(models) != 1:
        raise SystemExit(f"states from different models: {models}")
    mf = models.pop()
    states = states[:a.max]
    cfg, m = load_model(a.label, mf, dev)
    m.eval()
    extra = a.extra * a.L
    marks = sorted({int(c * a.L) for c in (1, 3, 13, 20, 40) if c * a.L <= extra} | {extra})
    t0 = time.monotonic()
    rows, warm_flips = replay_states(m, states, a.L, dev, extra, a.variants.split(","), a.batch, 2 * a.L, set(marks))
    res = dict(cmd="replay", label=a.label, model=mf, model_hash=file_hash(Path("runs") / f"nca-{a.label}" / mf),
               L=a.L, p=a.p, n_states=len(states), extra_rounds=extra, marks=marks, files=[f.name for f in files],
               warm_rounds=2 * a.L, warm_flips=warm_flips, diag_hash=DIAG_HASH, dep_hash=DEP_HASH, device=str(dev),
               seconds=round(time.monotonic() - t0, 1), rows=rows)
    jf = out / f"replay-{a.label}-L{a.L}-p{a.p:.4f}.json"
    tmp = jf.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, jf)
    for v in a.variants.split(","):
        r = [x for x in rows if x["variant"] == v]
        cl = [x["tclear"] for x in r if x["tclear"] >= 0]
        print(f"replay L={a.L} {v}: {len(r)} states, cleared {len(cl)} (median round {np.median(cl) if cl else '-'}), "
              f"logical error at clear {sum(x['w_at_clear'] for x in r if x['tclear'] >= 0)}, "
              f"re-opened {sum(x['reopen'] for x in r)}", flush=True)


# ------------------------------------------------------------------------------------------------ planted pairs

def pair_error(L, d, orient, i0, j0):
    """an error string of d links from site (i0, j0): straight along x, or a staircase x,y,x,y,... -> 2 defects."""
    e = torch.zeros(2, L, L, dtype=torch.bool)
    i, j = i0, j0
    for s in range(d):
        if orient == "x" or s % 2 == 0:
            e[0, i, j] ^= True
            i = (i + 1) % L
        else:
            e[1, i, j] ^= True
            j = (j + 1) % L
    return e, (i0, j0), (i, j)


def run_pairs_clean(model, L, ds, orients, memories, rounds, dev):
    xin = getattr(model, "xin", 2)
    cfgs = [(d, o, mem) for d in ds for o in orients for mem in memories]
    e = torch.stack([pair_error(L, d, o, L // 4, L // 4)[0] for d, o, _ in cfgs]).to(dev)
    B = len(cfgs)
    hw = warm_state(model, L, dev, 2 * L)[0] if "warm" in memories else None
    h = torch.cat([model.init_state(1, L, dev) if mem == "zero" else hw for _, _, mem in cfgs])
    corr = torch.zeros_like(e)
    prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
    lastflip = torch.zeros_like(e)
    ndef_tr, dist_tr = [], []
    tclear = [-1] * B
    wclear = [False] * B
    with torch.no_grad():
        for t in range(rounds):
            td = synd(e ^ corr)
            h, logits = model.round(h, observe(td, prev, lastflip, xin))
            flips = model.flips(model.greedy(logits))
            prev, lastflip = td, flips
            corr = corr ^ flips
            dd = e ^ corr
            sy = synd(dd)
            nd = sy.flatten(1).sum(1).cpu().tolist()
            w = winding(dd).cpu().tolist()
            dist = []
            for b in range(B):
                if nd[b] == 2:
                    pts = torch.nonzero(sy[b]).cpu().tolist()
                    dist.append(tdist(pts[0], pts[1], L))
                else:
                    dist.append(-1)
                if nd[b] == 0 and tclear[b] < 0:
                    tclear[b], wclear[b] = t + 1, bool(w[b])
            ndef_tr.append(nd)
            dist_tr.append(dist)
    nd = np.array(ndef_tr).T
    di = np.array(dist_tr).T
    rows = []
    for b, (d, o, mem) in enumerate(cfgs):
        steps = [x for x in di[b] if x >= 0]
        rows.append(dict(d=d, orient=o, memory=mem, tclear=tclear[b], logical_error_at_clear=wclear[b],
                         max_ndef=int(nd[b].max()), min_dist=int(min(steps)) if steps else -1,
                         max_dist=int(max(steps)) if steps else -1,
                         reopened=bool(tclear[b] >= 0 and nd[b][tclear[b]:].max() > 0),
                         dist_trace=[int(x) for x in di[b][:400]] + [int(x) for x in di[b][400::10]],
                         ndef_trace=[int(x) for x in nd[b][:400]] + [int(x) for x in nd[b][400::10]]))
    return rows


def run_pairs_noisy(model, L, d, p, runs, batch, seed, dev, noisy_rounds, clean_rounds, orient="x"):
    """runs with a planted pair (d > 0) or without (d = 0) under the SAME noise (same seed): background noise on data
    and readings for noisy_rounds, then clean_rounds noiseless; strict / left / matched at the end, matched after noise."""
    gen = torch.Generator().manual_seed(seed)
    xin = getattr(model, "xin", 2)
    res = dict(strict=[], left=[], matched=[], matched_noise_end=[])
    n = 0
    with torch.no_grad():
        while n < runs:
            B = min(batch, runs - n)
            e = torch.zeros(B, 2, L, L, dtype=torch.bool, device=dev)
            if d > 0:
                e = e ^ pair_error(L, d, orient, L // 4, L // 4)[0].to(dev).unsqueeze(0)
            corr = torch.zeros_like(e)
            prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
            lastflip = torch.zeros_like(e)
            h = model.init_state(B, L, dev)
            for t in range(noisy_rounds + clean_rounds):
                noisy = t < noisy_rounds
                if noisy:
                    e = e ^ (rnd(e.shape, gen, dev) < p)
                td = synd(e ^ corr)
                m = td ^ (rnd(td.shape, gen, dev) < p) if noisy else td
                h, logits = model.round(h, observe(m, prev, lastflip, xin))
                flips = model.flips(model.greedy(logits))
                prev, lastflip = m, flips
                corr = corr ^ flips
                if t + 1 == noisy_rounds:
                    dd = e ^ corr
                    res["matched_noise_end"].append(winding(dd ^ ideal_fix(dd)).cpu().numpy())
            dd = e ^ corr
            left = synd(dd).flatten(1).any(1)
            res["strict"].append((left | winding(dd)).cpu().numpy())
            res["left"].append(left.cpu().numpy())
            res["matched"].append(winding(dd ^ ideal_fix(dd)).cpu().numpy())
            n += B
    return {k: np.concatenate(v) for k, v in res.items()}


def cmd_pairs(a):
    os.chdir(HERE)
    from decoder_v7 import file_hash, load_model
    dev = torch.device(a.device)
    out = Path(a.out) / a.sub
    out.mkdir(parents=True, exist_ok=True)
    cfg, m = load_model(a.label, a.model, dev)
    m.eval()
    t0 = time.monotonic()
    ds = [int(x) for x in a.ds.split(",")]
    rows = run_pairs_clean(m, a.L, ds, a.orients.split(","), a.memories.split(","), a.rounds, dev)
    for r in rows:
        print(f"pair d={r['d']:2d} {r['orient']:4s} {r['memory']:4s}: cleared at round {r['tclear']}  "
              f"logical {r['logical_error_at_clear']}  dist min/max {r['min_dist']}/{r['max_dist']}  "
              f"max defects {r['max_ndef']}  reopened {r['reopened']}", flush=True)
    res = dict(cmd="pairs", label=a.label, model=a.model, model_hash=file_hash(Path("runs") / f"nca-{a.label}" / a.model),
               L=a.L, rounds=a.rounds, clean=rows, diag_hash=DIAG_HASH, dep_hash=DEP_HASH, device=str(dev))
    if a.bg_p > 0:
        ctrl = run_pairs_noisy(m, a.L, 0, a.bg_p, a.bg_runs, a.batch, a.seed, dev, 2 * a.L, 20 * a.L)
        res["noisy"] = dict(p=a.bg_p, runs=a.bg_runs, seed=a.seed, noisy_rounds=2 * a.L, clean_rounds=20 * a.L, rows=[])
        todo = [(0, "-", ctrl)] + [(d, o, None) for d in (int(x) for x in a.bg_ds.split(","))
                                   for o in a.noisy_orients.split(",")]
        for d, o, r in todo:
            if r is None:
                r = run_pairs_noisy(m, a.L, d, a.bg_p, a.bg_runs, a.batch, a.seed, dev, 2 * a.L, 20 * a.L, orient=o)
            row = dict(d=d, orient=o, **{k: int(v.sum()) for k, v in r.items()})
            if d > 0:
                row["strict_only_pair"] = int((r["strict"] & ~ctrl["strict"]).sum())
                row["strict_only_ctrl"] = int((~r["strict"] & ctrl["strict"]).sum())
                row["matched_only_pair"] = int((r["matched"] & ~ctrl["matched"]).sum())
                row["matched_only_ctrl"] = int((~r["matched"] & ctrl["matched"]).sum())
            res["noisy"]["rows"].append(row)
            print(f"noisy p={a.bg_p} d={d} {o}: {row}", flush=True)
    res["seconds"] = round(time.monotonic() - t0, 1)
    jf = out / f"pairs-{a.label}-L{a.L}{('-' + a.name) if a.name else ''}.json"
    tmp = jf.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, jf)


# ------------------------------------------------------------------------------------------------ summary

def cmd_summary(a):
    out = Path(a.out) / a.sub
    L_ = [f"# Diagnostics, {a.sub} (SIMULATED). Full evolution of every run (no shortcut).", "",
          "strict = defects left or logical error; left = defects left; matched = local evolution + global MWPM readout "
          "on the true leftover syndrome; lake = Lake-style (stop when clean, up to 2L; unfinished runs scored by their "
          "winding, NOT as failures). cL = after c*L noiseless rounds (0 = right after the last noisy round).", ""]
    groups = {}
    for f in sorted(out.glob("score-*.json")):
        r = json.loads(f.read_text())
        for mf, c in r["models"].items():
            g = groups.setdefault((r["label"], mf, r["L"], r["p"]), dict(files=[], hashes=set(), c={}))
            g["files"].append(f.name)
            g["hashes"].add((c["model_hash"], r["diag_hash"], r["dep_hash"]))
            for k, v in c.items():
                if isinstance(v, int):
                    g["c"][k] = g["c"].get(k, 0) + v
    bad = [k for k, g in groups.items() if len(g["hashes"]) != 1]
    curve = {}
    for (label, mf, L, p), g in groups.items():
        curve.setdefault((label, mf), set()).update(g["hashes"])
    bad += [k for k, h in curve.items() if len(h) != 1]
    if bad:
        raise SystemExit(f"SUMMARY REFUSED: mixed model/code hashes in {bad}")
    L_ += ["## Scores (rate, 95% Wilson interval)", "",
           "| model | L | p | runs | strict 0L | strict 2L | strict 20L | left 20L | matched 0L | matched 2L | matched 20L "
           "| lake (unfinished at 2L) | reopen | silent | shortcut disagree 20L |", "|" + "---|" * 15]
    def fmt(k, n):
        if k is None:
            return "-"
        return f"{k} ({k / n:.1e} [{wilson(k, n)[0]:.1e}, {wilson(k, n)[1]:.1e}])"
    for (label, mf, L, p), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][3], kv[0][2])):
        c, n = g["c"], g["c"]["n"]
        L_.append(f"| {label} [{mf}] | {L} | {100 * p:.2f}% | {n} | {fmt(c.get('strict_0L'), n)} | "
                  f"{fmt(c.get('strict_2L'), n)} | {fmt(c.get('strict_20L'), n)} | {c.get('left_20L', '-')} | "
                  f"{fmt(c.get('matched_0L'), n)} | {fmt(c.get('matched_2L'), n)} | {fmt(c.get('matched_20L'), n)} | "
                  f"{fmt(c['lake'], n)} ({c['lake_unfinished']}) | {c['reopen']} | "
                  f"{c['silent']} | {c.get('shortcut_disagree_20L', '-')} |")
    pairs = {}
    for f in sorted(out.glob("score-*.json")):
        r = json.loads(f.read_text())
        if "paired" in r:
            key = (r["L"], r["p"], r["paired"]["A"], r["paired"]["B"])
            P = pairs.setdefault(key, dict(ident=True))
            P["ident"] &= r["paired_noise_identical"]
            for k in ("strict_2L", "strict_20L", "matched_0L", "matched_20L"):
                for kk, v in r["paired"].get(k, {}).items():
                    P[(k, kk)] = P.get((k, kk), 0) + v
    if pairs:
        L_ += ["", "## Paired comparison on identical noise (A vs B): both fail / only A / only B", "",
               "| L | p | A | B | noise identical | strict 2L | strict 20L | matched 0L | matched 20L |", "|" + "---|" * 9]
        for (L, p, A, B), P in sorted(pairs.items()):
            cell = lambda k: (f"{P[(k, 'both')]} / {P[(k, 'only_A')]} / {P[(k, 'only_B')]}"  # noqa: E731
                              if (k, "both") in P else "-")
            L_.append(f"| {L} | {100 * p:.2f}% | {A} | {B} | {P['ident']} | {cell('strict_2L')} | {cell('strict_20L')} | "
                      f"{cell('matched_0L')} | {cell('matched_20L')} |")
    for f in sorted(out.glob("replay-*.json")):
        r = json.loads(f.read_text())
        L_ += ["", f"## Replay of runs unfinished at 6L, L={r['L']}, p={100 * r['p']:.2f}% ({r['n_states']} states, model {r['model']}, "
               f"+{r['extra_rounds']} noiseless rounds; warm-up flipped {r['warm_flips']} links on a clean lattice)", "",
               "| memory | cleared | median rounds to clear | logical error at clear | re-opened | start: defects (median) "
               "| start: nearest-neighbour distance max (median) | start: MWPM weight (median) |", "|" + "---|" * 8]
        for v in ("original", "zero", "warm"):
            rr = [x for x in r["rows"] if x["variant"] == v]
            if not rr:
                continue
            cl = [x["tclear"] for x in rr if x["tclear"] >= 0]
            L_.append(f"| {v} | {len(cl)}/{len(rr)} | {np.median(cl) if cl else '-'} | "
                      f"{sum(x['w_at_clear'] for x in rr if x['tclear'] >= 0)} | {sum(x['reopen'] for x in rr)} | "
                      f"{np.median([x['start']['ndef'] for x in rr])} | {np.median([x['start']['nn_max'] for x in rr])} | "
                      f"{np.median([x['start']['mwpm_weight'] for x in rr])} |")
        orig = {x["shot"]: x for x in r["rows"] if x["variant"] == "original"}
        for v in ("zero", "warm"):
            vv = {x["shot"]: x for x in r["rows"] if x["variant"] == v}
            if vv:
                both = [s for s in orig if s in vv]
                a_ = sum(1 for s in both if orig[s]["tclear"] < 0 <= vv[s]["tclear"])
                b_ = sum(1 for s in both if vv[s]["tclear"] < 0 <= orig[s]["tclear"])
                L_.append(f"\npaired with original: cleared only with {v} memory {a_}, only with original memory {b_}")
        stuck = [x for x in r["rows"] if x["variant"] == "original" and x["tclear"] < 0]
        if stuck:
            L_.append(f"\nnever cleared with original memory: {len(stuck)}; their start defects (median) "
                      f"{np.median([x['start']['ndef'] for x in stuck])}, MWPM weight (median) "
                      f"{np.median([x['start']['mwpm_weight'] for x in stuck])}")
    for f in sorted(out.glob("pairs-*.json")):
        r = json.loads(f.read_text())
        L_ += ["", f"## Planted pair ({f.stem}), L={r['L']}, no noise, {r['rounds']} rounds, model {r['model']}", "",
               "| d | orientation | memory | cleared at round | logical error | closest / farthest | max defects | re-opened |",
               "|" + "---|" * 8]
        for x in r["clean"]:
            L_.append(f"| {x['d']} | {x['orient']} | {x['memory']} | {x['tclear']} | {x['logical_error_at_clear']} | "
                      f"{x['min_dist']} / {x['max_dist']} | {x['max_ndef']} | {x['reopened']} |")
        if "noisy" in r:
            nz = r["noisy"]
            L_ += ["", f"Planted pair under background noise p={nz['p']} ({nz['runs']} runs, same noise as the control "
                   f"d=0; {nz['noisy_rounds']} noisy + {nz['clean_rounds']} noiseless rounds)", "",
                   "| d | orientation | strict | left | matched | matched at noise end | strict only with pair / only control |",
                   "|" + "---|" * 7]
            for x in nz["rows"]:
                L_.append(f"| {x['d']} | {x.get('orient', 'x')} | {x['strict']} | {x['left']} | {x['matched']} | {x['matched_noise_end']} | "
                          f"{x.get('strict_only_pair', '-')} / {x.get('strict_only_ctrl', '-')} |")
    text = "\n".join(L_) + "\n"
    (out / f"{a.sub}-summary.md").write_text(text)
    print(text)


# ------------------------------------------------------------------------------------------------ checks

class Toy(torch.nn.Module):
    """known-answer rules with the decoder interface. idle: never flips. waitflip: after `trigger` rounds flips every
    x-link once (on an odd torus this changes no syndrome but flips the logical class: the referee's counterexample)."""

    def __init__(self, mode, trigger=0):
        super().__init__()
        self.mode, self.trigger, self.xin, self.C = mode, trigger, 2, 1

    def init_state(self, B, L, dev):
        return torch.zeros(B, 1, L, L, device=dev)

    def round(self, h, x):
        h = h + 1
        lg = -torch.ones(h.shape[0], 2, h.shape[2], h.shape[3], device=h.device)
        if self.mode == "waitflip":
            fire = (h[:, 0, 0, 0] == self.trigger).view(-1, 1, 1)
            lg[:, 0] = torch.where(fire, torch.ones_like(lg[:, 0]), lg[:, 0])
        return h, lg

    def greedy(self, lg):
        return lg > 0

    def flips(self, a):
        return a


def random_rule(seed):
    torch.manual_seed(seed)
    r = Rule(C=4, M=2, K=2, hidden=8)
    for head in r.act:
        torch.nn.init.constant_(head[-1].bias, 0.0)          # make it act, so it moves defects around
    r.xin = 2
    return r.eval()


def cmd_check(a):
    from decoder_v7 import evaluate_multi
    sys.path.insert(0, str(HERE / "submit"))
    from submit_eval import evaluate_long
    dev = torch.device("cpu")
    ok = True

    def report(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)

    # 1. referee counterexample: shortcut misses a silent logical change; full evolution sees it
    L = 3
    _, arr, _, _ = run_full(Toy("waitflip", trigger=L + 8 * L), L, 0.0, 4, 4, 1, dev)
    c = counts_from(arr)
    report("1 waitflip: full evolution fails 4/4 at 10L, 20L", c["strict_10L"] == 4 and c["strict_20L"] == 4, str(c))
    report("1b waitflip: shortcut emulation says 0/4 and disagrees on 4", c["shortcut_strict_20L"] == 0 and
           c["shortcut_disagree_20L"] == 4)
    report("1c waitflip: counted as 'silent' logical change while clean", c["silent"] == 4)
    _, arr, _, _ = run_full(Toy("idle"), L, 0.0, 4, 4, 1, dev)
    c = counts_from(arr)
    report("1d idle rule, no noise: nothing fails, nothing silent", c["strict_20L"] == 0 and c["silent"] == 0 and
           c["shortcut_disagree_20L"] == 0)

    # 2. same loop and random numbers as decoder_v7.evaluate_multi; shortcut emulation == evaluate_long
    m = random_rule(3)
    L, p, shots, batch, seed = 9, 0.03, 24, 8, 77
    n, arr, nh, saved = run_full(m, L, p, shots, batch, seed, dev, save_max=24)
    c = counts_from(arr)
    _, ref = evaluate_multi(m, L, p, shots, batch, seed, dev, cleans=CKS)
    same = all(c[f"strict_{k}L"] == ref[k]["fixed"] and c[f"left_{k}L"] == ref[k]["leftover"] and
               c[f"matched_{k}L"] == ref[k]["matched"] for k in CKS)
    report("2 full evolution == decoder_v7.evaluate_multi (all checkpoints)", same,
           str({k: (c[f'strict_{k}L'], ref[k]['fixed']) for k in CKS}))
    _, lo = evaluate_long(m, L, p, shots, batch, seed, dev)
    report("2b shortcut emulation == submit_eval.evaluate_long at 10L, 20L",
           c["shortcut_strict_10L"] == lo[10]["fixed"] and c["shortcut_strict_20L"] == lo[20]["fixed"],
           f"{c['shortcut_strict_20L']} vs {lo[20]['fixed']}")
    report("2c test rule leaves runs unfinished at 6L (so the next checks mean something)", len(saved) > 3,
           f"{len(saved)} saved")

    # 3. paired noise: same seed -> identical noise for two different rules; different seed -> different
    _, _, nh2, _ = run_full(random_rule(5), L, p, shots, batch, seed, dev)
    _, _, nh3, _ = run_full(m, L, p, shots, batch, seed + 1, dev)
    report("3 same seed, different rule: identical noise", nh == nh2)
    report("3b different seed: different noise (negative control)", nh != nh3)

    # 4. replay with original memory = exact continuation of the full run (saved at 6L, i.e. after 7L rounds;
    #    the 10L / 20L checkpoints are 4L / 14L rounds later). Compared on the full correction, not only on scores.
    _, _, _, saved20 = run_full(m, L, p, shots, batch, seed, dev, save_max=24, save_at=20)
    at20 = {s["shot"]: s["corr"] for s in saved20}
    marks = {4 * L, 14 * L}
    rows, _, fin = replay_states(m, saved, L, dev, 14 * L, ["original", "zero", "warm"], 5, 2 * L, marks,
                                 return_final=True)
    nv = len(saved)
    orig, zero = rows[:nv], rows[nv:2 * nv]
    ok4 = all(r["track"][str(4 * L)]["left"] == bool(arr["left_10"][r["shot"]]) and
              r["track"][str(14 * L)]["left"] == bool(arr["left_20"][r["shot"]]) and
              r["track"][str(14 * L)]["w"] == bool(arr["w_20"][r["shot"]]) for r in orig)
    ok4 &= all(r["shot"] in at20 and torch.equal(fin[i], at20[r["shot"]]) for i, r in enumerate(orig))
    report("4 replay(original) reproduces the full run exactly (scores at 10L, 20L and the whole correction at 20L)",
           ok4 and len(at20) == nv, f"{nv} states")
    # 4b/4c need a rule whose memory matters for a long time (the small random rule forgets within a few ticks):
    # waitflip counts rounds in its memory and flips every x-link at round 15L. Continued with its own memory it fires
    # 8L rounds into the replay; with zero memory it would fire only after 15L > 14L, i.e. never in the window.
    tw = Toy("waitflip", trigger=15 * L)
    _, arr_w, _, sv_w = run_full(tw, L, p, 8, 8, seed, dev, save_max=8)
    _, _, _, sv_w20 = run_full(tw, L, p, 8, 8, seed, dev, save_max=8, save_at=20)
    ref_w = {s["shot"]: s["corr"] for s in sv_w20}
    rw, _, fw = replay_states(tw, sv_w, L, dev, 14 * L, ["original", "zero"], 8, 2 * L, marks, return_final=True)
    k = len(sv_w)
    okw = k > 0 and all(torch.equal(fw[i], ref_w[rw[i]["shot"]]) for i in range(k))
    diff = sum(1 for i in range(k) if not torch.equal(fw[i], fw[k + i]))
    report("4b memory-dependent rule: original memory reproduces the run, zero memory changes it", okw and diff == k,
           f"{diff}/{k} differ")
    broken = [dict(s, h=torch.zeros_like(s["h"])) for s in sv_w]
    _, _, fb = replay_states(tw, broken, L, dev, 14 * L, ["original"], 8, 2 * L, marks, return_final=True)
    nb = sum(1 for i in range(k) if not torch.equal(fb[i], ref_w[rw[i]["shot"]]))
    report("4c negative control: states with wiped memory do NOT reproduce the run", nb == k, f"{nb}/{k} differ")
    # 2d: equality with evaluate_multi on a case with mixed outcomes (idle rule, low noise)
    _, arr_i, _, _ = run_full(Toy("idle"), 9, 0.0008, 40, 8, 21, dev)
    ci = counts_from(arr_i)
    _, ri = evaluate_multi(Toy("idle"), 9, 0.0008, 40, 8, 21, dev, cleans=CKS)
    report("2d full evolution == evaluate_multi on mixed outcomes (idle rule, p = 0.08%)",
           all(ci[f"strict_{c}L"] == ri[c]["fixed"] and ci[f"matched_{c}L"] == ri[c]["matched"] for c in CKS) and
           0 < ci["strict_20L"] < 40 and ci["matched_20L"] < ci["strict_20L"],
           f"strict {ci['strict_20L']}/40, matched {ci['matched_20L']}")

    # 5. Lake-style score: idle rule never cleans -> scored by its winding at 2L; no-noise -> clean at once
    _, arr, _, _ = run_full(Toy("idle"), 9, 0.05, 16, 16, 5, dev)
    c = counts_from(arr)
    exp_unf = int(arr["left_2"].sum())
    report("5 Lake-style: unfinished runs (idle rule) scored by winding, counted as unfinished",
           c["lake_unfinished"] == exp_unf and c["lake"] == int(arr["w_2"].sum()), f"{c['lake']} {c['lake_unfinished']}")
    _, arr, _, _ = run_full(Toy("idle"), 9, 0.0, 4, 4, 5, dev)
    report("5b no noise: clean at round 0, lake 0", (arr["first_clean"] == 0).all() and counts_from(arr)["lake"] == 0)

    # 6. planted pairs: exactly two defects at the intended distance, no winding, both orientations
    L = 111
    okp = True
    for d in (2, 4, 8, 12, 16, 24, 32, 48):
        for o in ("x", "diag"):
            e, s, t = pair_error(L, d, o, L // 4, L // 4)
            sy = synd(e.unsqueeze(0))[0]
            pts = sorted(map(tuple, torch.nonzero(sy).tolist()))
            dx, dy = (t[0] - s[0]) % L, (t[1] - s[1]) % L
            want = (d, 0) if o == "x" else ((d + 1) // 2, d // 2)
            okp &= (len(pts) == 2 and tdist(pts[0], pts[1], L) == d and not bool(winding(e.unsqueeze(0))[0])
                    and sorted([s, t]) == pts and (dx, dy) == want)
    report("6 planted pairs: two defects at distance d, intended direction, no winding (x and staircase, d = 2..48)", okp)
    e, _, _ = pair_error(11, 11, "x", 0, 0)
    report("6b negative control: a string around the torus leaves no defects but winds",
           int(synd(e.unsqueeze(0)).sum()) == 0 and bool(winding(e.unsqueeze(0))[0]))
    r = run_pairs_clean(Toy("idle"), 31, [4, 8], ["x", "diag"], ["zero"], 20, dev)
    report("6c idle rule on planted pairs: never cleared, distance stays d",
           all(x["tclear"] < 0 and x["min_dist"] == x["max_dist"] == x["d"] for x in r))

    # 7. noisy pair vs control: same seed -> identical noise; d = 0 equals a plain run of the same seed
    r0 = run_pairs_noisy(m, 9, 0, 0.03, 16, 8, 11, dev, 9, 18)
    r0b = run_pairs_noisy(m, 9, 0, 0.03, 16, 8, 11, dev, 9, 18)
    report("7 noisy pair control is reproducible", all((r0[k] == r0b[k]).all() for k in r0))
    _, ref = evaluate_multi(m, 9, 0.03, 16, 8, 11, dev, cleans=(2,))
    report("7b control (d = 0, L noisy + 2L clean) == evaluate_multi", int(r0["strict"].sum()) == ref[2]["fixed"])
    print("DIAG CHECK OK" if ok else "DIAG CHECK FAILED", flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--label", default="v10-big"); s.add_argument("--models", default="model_last.pt")
    s.add_argument("--L", type=int, required=True); s.add_argument("--p", type=float, required=True)
    s.add_argument("--shots", type=int, required=True); s.add_argument("--batch", type=int, default=32)
    s.add_argument("--seed", type=int, required=True); s.add_argument("--save", type=int, default=0)
    s.add_argument("--out", required=True); s.add_argument("--device", default="cuda")
    s.add_argument("--sub", default="diag1"); s.add_argument("--cks", default="0,2,6,10,20")
    r = sub.add_parser("replay")
    r.add_argument("--label", default="v10-big"); r.add_argument("--L", type=int, required=True)
    r.add_argument("--p", type=float, required=True)
    r.add_argument("--variants", default="original,zero,warm"); r.add_argument("--extra", type=int, default=40)
    r.add_argument("--max", type=int, default=200); r.add_argument("--batch", type=int, default=32)
    r.add_argument("--out", required=True); r.add_argument("--device", default="cuda")
    r.add_argument("--sub", default="diag1")
    q = sub.add_parser("pairs")
    q.add_argument("--label", default="v10-big"); q.add_argument("--model", default="model_last.pt")
    q.add_argument("--L", type=int, default=111); q.add_argument("--ds", default="2,4,8,12,16,24,32,48")
    q.add_argument("--orients", default="x,diag"); q.add_argument("--memories", default="zero,warm")
    q.add_argument("--rounds", type=int, default=2000); q.add_argument("--bg-p", type=float, default=0.0)
    q.add_argument("--bg-ds", default="4,8,16,32"); q.add_argument("--bg-runs", type=int, default=200)
    q.add_argument("--batch", type=int, default=32); q.add_argument("--seed", type=int, default=9091)
    q.add_argument("--out", required=True); q.add_argument("--device", default="cuda")
    q.add_argument("--sub", default="diag1"); q.add_argument("--name", default="")
    q.add_argument("--noisy-orients", default="x")
    m = sub.add_parser("summary")
    m.add_argument("--out", required=True); m.add_argument("--sub", default="diag1")
    sub.add_parser("check")
    a = ap.parse_args()
    {"score": cmd_score, "replay": cmd_replay, "pairs": cmd_pairs, "summary": cmd_summary, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
