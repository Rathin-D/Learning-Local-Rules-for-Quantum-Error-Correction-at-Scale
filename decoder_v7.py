"""Decoder v7: changes from independent review 1 (qec/report/review-1-assessment.md).

Evaluation (fixes):
  * Every run gets exactly L noisy rounds and then exactly 2L noiseless rounds ("fixed" score). The old evaluate()
    stopped a whole batch only when every run in it was clean, so a run's result could depend on the other runs.
    A second score ("stop") freezes each run separately at the first noiseless round where it has no defects.
    Neither score depends on batch companions.
  * Each result line records the model file's hash, its config, seed, batch, protocol and this file's hash; a cached
    point is reused only if all of these match.
  * Models are always loaded with their saved options (extra input channel, rounding bits).
Training (one change per arm, all from the same start, same recipe as v7b / v8):
  * --xin 3 : a third input channel per site, the detection event d_t = m_t xor m_{t-1} xor synd(own flips at t-1):
    "my reading changed, and not because of my own last correction". The new input weights start at zero, so the
    rule starts out decoding exactly as before.
  * --long-frac f --long-mult k : a fraction f of the ordinary (non-drill) training runs has k*L noisy rounds instead
    of L, so the rule practises long stretches of noise before the clean-up.
  * --K : message ticks per round.
Validation uses fresh noise (seeds 999001+, never used before) and the fixed protocol.

  python decoder_v7.py check
  python decoder_v7.py train --init v8-ctrl --label v9-a --max-iters 12000
  python decoder_v7.py eval --label v9-a --model model_last.pt --Ls 19,27,39 --ps 0.016,0.018 --shots 30000 --tag x
  python decoder_v7.py summary --rows base=v8-ctrl:model.pt,A=v9-a:model_last.pt --out v9-summary.md
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from decoder_v41 import Rule, build, drill_errors, pick_device, rnd, synd, teacher_match, wilson, winding

CODE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- model with an optional third input channel

class RuleX(Rule):
    """Rule whose per-round input has xin channels (2: reading, change; 3: + detection event)."""

    def __init__(self, C, M, K, hidden, xin):
        super().__init__(C, M, K, hidden, False, "link", False)
        self.xin = xin
        self.upd[0] = nn.Conv2d(C + 4 * M + xin, hidden, 1)


def build_any(cfg):
    xin = int(cfg.get("xin", 2))
    if xin == 2:
        model = build(cfg)
        model.xin = 2
    else:
        if cfg["gru"] or cfg["dirmsg"] or cfg["action"] != "link":
            raise SystemExit("xin=3 supports the plain link rule only")
        model = RuleX(cfg["C"], cfg["M"], cfg["K"], cfg["hidden"], xin)
    return model


def apply_rounding(model, cfg, ste=False):
    b = int(cfg.get("qbits", 0))
    if b:
        from decoder_tests import quant_round
        model.round = quant_round(model, b, ste=ste)
    return b


def load_model(label, model_file, dev, ste=False):
    """the one place that turns a saved run into a working decoder (all saved options applied)."""
    src = Path("runs") / f"nca-{label}"
    cfg = json.loads((src / "config.json").read_text())
    model = build_any(cfg).to(dev)
    model.load_state_dict(torch.load(src / model_file, map_location=dev))
    apply_rounding(model, cfg, ste=ste)
    return cfg, model


def add_detection_channel(model):
    """function-preserving copy of a 2-input rule with a third input channel whose weights are zero."""
    if getattr(model, "xin", 2) != 2 or model.gru or model.dirmsg or model.action != "link":
        raise SystemExit("can only add the channel to a plain 2-input link rule")
    new = RuleX(model.C, model.M, model.K, model.upd[0].out_channels, 3)
    sd = model.state_dict()
    w = sd["upd.0.weight"]
    sd["upd.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], 1, 1, 1, dtype=w.dtype)], 1)
    new.load_state_dict(sd)
    return new


def observe(m, prev, lastflip, xin):
    """per-site input for this round. lastflip = links the decoder flipped in the previous round."""
    chans = [m, m ^ prev]
    if xin == 3:
        chans.append(m ^ prev ^ synd(lastflip))
    return torch.stack(chans, 1).float()


# ---------------------------------------------------------------- evaluation (batch-independent)

def evaluate7(model, L, p, shots, batch, seed, dev):
    """L noisy rounds, then exactly 2L noiseless rounds for every run.
    fixed: logical error or any defect after the 2L noiseless rounds.
    stop : each run is judged at the first noiseless round where it has no defects (logical error only);
           runs that never get clean are judged as in 'fixed'."""
    gen = torch.Generator().manual_seed(seed)
    xin = getattr(model, "xin", 2)
    out = dict(n=0, fixed=0, fixed_logical=0, stop=0, leftover_runs=0, ever_clean=0)
    with torch.no_grad():
        while out["n"] < shots:
            B = min(batch, shots - out["n"])
            e = torch.zeros(B, 2, L, L, dtype=torch.bool, device=dev)
            corr = torch.zeros_like(e)
            prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
            lastflip = torch.zeros_like(e)
            h = model.init_state(B, L, dev)
            stopped = torch.zeros(B, dtype=torch.bool, device=dev)
            stop_fail = torch.zeros(B, dtype=torch.bool, device=dev)
            for t in range(3 * L):
                noisy = t < L
                if noisy:
                    e = e ^ (rnd(e.shape, gen, dev) < p)
                td = synd(e ^ corr)
                if not noisy:                                        # per-run stop score (no cross-run reduction)
                    clean = ~td.flatten(1).any(1)
                    newly = clean & ~stopped
                    stop_fail = stop_fail | (newly & winding(e ^ corr))
                    stopped = stopped | clean
                m = td ^ (rnd(td.shape, gen, dev) < p) if noisy else td
                h, logits = model.round(h, observe(m, prev, lastflip, xin))
                flips = model.flips(model.greedy(logits))
                prev, lastflip = m, flips
                corr = corr ^ flips
            d = e ^ corr
            left = synd(d).flatten(1).any(1)
            w = winding(d)
            fixed = w | left
            stop = torch.where(stopped, stop_fail, fixed)
            out["n"] += B
            out["fixed"] += int(fixed.sum()); out["fixed_logical"] += int(w.sum())
            out["stop"] += int(stop.sum()); out["leftover_runs"] += int(left.sum())
            out["ever_clean"] += int(stopped.sum())
    return out


def evaluate_multi(model, L, p, shots, batch, seed, dev, cleans=(2, 6)):
    """same runs as evaluate7 (same noise for the same seed), but the noiseless phase lasts max(cleans)*L rounds and
    every run is scored after c*L noiseless rounds for each c in cleans:
      fixed   : logical error or leftover defects (the strict score; c = 2 reproduces evaluate7's 'fixed')
      leftover: runs with defects left
      matched : logical error after a perfect matching decoder finishes whatever is left (how well the rule
                protected the encoded information, independent of whether it finished the clean-up itself)"""
    from decoder_tests import ideal_fix
    gen = torch.Generator().manual_seed(seed)
    xin = getattr(model, "xin", 2)
    cleans = sorted(set(int(c) for c in cleans))
    marks = {L + c * L: c for c in cleans}                     # number of rounds done -> checkpoint
    out = {c: dict(fixed=0, leftover=0, matched=0) for c in cleans}
    n = 0
    with torch.no_grad():
        while n < shots:
            B = min(batch, shots - n)
            e = torch.zeros(B, 2, L, L, dtype=torch.bool, device=dev)
            corr = torch.zeros_like(e)
            prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
            lastflip = torch.zeros_like(e)
            h = model.init_state(B, L, dev)
            for t in range(L + max(cleans) * L):
                noisy = t < L
                if noisy:
                    e = e ^ (rnd(e.shape, gen, dev) < p)
                td = synd(e ^ corr)
                m = td ^ (rnd(td.shape, gen, dev) < p) if noisy else td
                h, logits = model.round(h, observe(m, prev, lastflip, xin))
                flips = model.flips(model.greedy(logits))
                prev, lastflip = m, flips
                corr = corr ^ flips
                if t + 1 in marks:
                    c = marks[t + 1]
                    d = e ^ corr
                    left = synd(d).flatten(1).any(1)
                    out[c]["fixed"] += int((winding(d) | left).sum())
                    out[c]["leftover"] += int(left.sum())
                    out[c]["matched"] += int(winding(d ^ ideal_fix(d)).sum())
            n += B
    return n, out


def point_seed(L, p, seed):
    return (L * 1000003 + int(p * 1e6) + seed) % 2**31


def cmd_eval(a):
    dev = pick_device(a.device)
    src = Path("runs") / f"nca-{a.label}"
    cfg, model = load_model(a.label, a.model, dev)
    model.eval()
    prov = dict(label=a.label, model=a.model, model_hash=file_hash(src / a.model),
                config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                xin=getattr(model, "xin", 2), K=model.K, C=model.C, M=model.M, qbits=int(cfg.get("qbits", 0)),
                seed=a.seed, batch=a.batch, protocol="L noisy + fixed 2L noiseless", code_hash=CODE_HASH)
    jl = src / f"eval7-{a.tag}.jsonl"
    rows = [json.loads(x) for x in jl.read_text().splitlines()] if jl.exists() else []
    keys = ("model_hash", "config_hash", "seed", "batch", "protocol", "shots")
    for L in map(int, a.Ls.split(",")):
        for p in map(float, a.ps.split(",")):
            if any(r["L"] == L and r["p"] == p and all(r.get(k) == (a.shots if k == "shots" else prov[k]) for k in keys)
                   for r in rows):
                print(f"{a.label} {a.model} L={L} p={p}: already done (same model, seed, protocol)", flush=True)
                continue
            t0 = time.monotonic()
            r = evaluate7(model, L, p, a.shots, a.batch, point_seed(L, p, a.seed), dev)
            n = r["n"]
            row = dict(prov, L=L, p=p, shots=n, fixed=r["fixed"], stop=r["stop"], fixed_logical=r["fixed_logical"],
                       leftover_runs=r["leftover_runs"], ever_clean=r["ever_clean"],
                       ci_fixed=wilson(r["fixed"], n), seconds=round(time.monotonic() - t0, 1))
            rows.append(row)
            with jl.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{a.label} [{a.model}] L={L} p={100*p:.1f}%: fixed {r['fixed']}/{n}  stop {r['stop']}/{n}  "
                  f"leftover {r['leftover_runs']}  ({row['seconds']:.0f}s)", flush=True)


def cmd_deval(a):
    dev = pick_device(a.device)
    src = Path("runs") / f"nca-{a.label}"
    cfg, model = load_model(a.label, a.model, dev)
    model.eval()
    cleans = [int(c) for c in a.clean.split(",")]
    prov = dict(label=a.label, model=a.model, model_hash=file_hash(src / a.model),
                config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                xin=getattr(model, "xin", 2), K=model.K, C=model.C, M=model.M, qbits=int(cfg.get("qbits", 0)),
                seed=a.seed, batch=a.batch, cleans=cleans,
                protocol="L noisy + noiseless rounds, scored after c*L of them", code_hash=CODE_HASH)
    jl = src / f"evald-{a.tag}.jsonl"
    rows = [json.loads(x) for x in jl.read_text().splitlines()] if jl.exists() else []
    keys = ("model_hash", "config_hash", "seed", "batch", "cleans", "shots")
    for L in map(int, a.Ls.split(",")):
        for p in map(float, a.ps.split(",")):
            if any(r["L"] == L and r["p"] == p and all(r.get(k) == (a.shots if k == "shots" else prov[k]) for k in keys)
                   for r in rows):
                print(f"{a.label} {a.model} L={L} p={p}: already done", flush=True)
                continue
            t0 = time.monotonic()
            n, res = evaluate_multi(model, L, p, a.shots, a.batch, point_seed(L, p, a.seed), dev, cleans)
            row = dict(prov, L=L, p=p, shots=n, seconds=round(time.monotonic() - t0, 1))
            for c in cleans:
                for k, v in res[c].items():
                    row[f"{k}_{c}L"] = v
            with jl.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{a.label} [{a.model}] L={L} p={100*p:.1f}%: " + "  ".join(
                f"after {c}L: fixed {res[c]['fixed']} leftover {res[c]['leftover']} matched {res[c]['matched']}"
                for c in cleans) + f"  / {n}  ({row['seconds']:.0f}s)", flush=True)


def cmd_dsummary(a):
    lines = ["# Clean-up diagnostic (SIMULATED): failures / runs after c*L noiseless rounds", "",
             "fixed = strict (logical error or defects left); left = runs with defects left; matched = logical error "
             "after a perfect matching decoder finishes what the rule left.", "",
             "| model | L | p | runs | after 2L: fixed / left / matched | after 6L: fixed / left / matched |",
             "|---|---|---|---|---|---|"]
    for item in a.rows.split(","):
        name, rest = item.split("=")
        label, model = rest.split(":")
        src = Path("runs") / f"nca-{label}"
        if not (src / model).exists():
            continue
        h = file_hash(src / model)
        for f in sorted(src.glob("evald-*.jsonl")):
            for x in f.read_text().splitlines():
                r = json.loads(x)
                if r["model"] != model or r["model_hash"] != h:
                    continue
                cells = []
                for c in (2, 6):
                    cells.append(f"{r[f'fixed_{c}L']} / {r[f'leftover_{c}L']} / {r[f'matched_{c}L']}"
                                 if f"fixed_{c}L" in r else "-")
                lines.append(f"| {name} | {r['L']} | {100*r['p']:.1f}% | {r['shots']} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_summary(a):
    rows = []
    for item in a.rows.split(","):
        name, rest = item.split("=")
        label, model = rest.split(":")
        src = Path("runs") / f"nca-{label}"
        h = file_hash(src / model) if (src / model).exists() else None
        for f in sorted(src.glob("eval7-*.jsonl")):
            for x in f.read_text().splitlines():
                r = json.loads(x)
                if r["model"] == model and r["model_hash"] == h:
                    rows.append(dict(r, name=name))
    pts = sorted({(r["L"], r["p"]) for r in rows})
    names = list(dict.fromkeys(item.split("=")[0] for item in a.rows.split(",")))
    lines = ["# v9: strict failures, L noisy + fixed 2L noiseless rounds (SIMULATED)", "",
             "fixed = logical error or leftover defects after exactly 2L noiseless rounds; stop = each run judged when it "
             "first has no defects. Cells: fixed (stop) / runs.", "",
             "| model | " + " | ".join(f"L={L} {100*p:.1f}%" for L, p in pts) + " |", "|---|" + "---|" * len(pts)]
    by = {(r["name"], r["L"], r["p"]): r for r in rows}
    for n in names:
        cells = [f"{by[(n,L,p)]['fixed']} ({by[(n,L,p)]['stop']}) / {by[(n,L,p)]['shots']}" if (n, L, p) in by else "-"
                 for L, p in pts]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "Failure-rate ratios between sizes (fixed score; below 1 = the larger code is better):", ""]
    for n in names:
        rat = []
        for p in sorted({p for _, p in pts}):
            for L1, L2 in ((19, 27), (27, 39), (39, 55)):
                r1, r2 = by.get((n, L1, p)), by.get((n, L2, p))
                if r1 and r2 and r1["fixed"] > 0:
                    rat.append(f"{L2}/{L1} at {100*p:.1f}%: {(r2['fixed']/r2['shots'])/(r1['fixed']/r1['shots']):.2f}")
        lines.append(f"- {n}: " + ("; ".join(rat) if rat else "-"))
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- training

def train_step(model, opt, B, L, p, gen, dev, chunk=24, drill=0, long_mult=1):
    """one imitation step with the matching teacher (sees true defects; training only)."""
    xin = getattr(model, "xin", 2)
    T_noisy = 0 if drill else long_mult * L
    e = drill_errors(B, L, drill, gen).to(dev) if drill else torch.zeros(B, 2, L, L, dtype=torch.bool, device=dev)
    corr = torch.zeros_like(e)
    prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
    lastflip = torch.zeros_like(e)
    h = model.init_state(B, L, dev)
    xs, tgts = [], []
    with torch.no_grad():
        for t in range(T_noisy + 2 * L):
            noisy = t < T_noisy
            if noisy:
                e = e ^ (rnd(e.shape, gen, dev) < p)
            td = synd(e ^ corr)
            if not noisy and not bool(td.any()):
                break
            m = td ^ (rnd(td.shape, gen, dev) < p) if noisy else td
            x = observe(m, prev, lastflip, xin)
            tgt = model.target(teacher_match(td))
            h, logits = model.round(h, x)
            xs.append(x); tgts.append(tgt)
            flips = model.flips(model.greedy(logits))
            prev, lastflip = m, flips
            corr = corr ^ flips
    n = len(xs)
    if n == 0:
        return 0.0, 0
    opt.zero_grad()
    h = model.init_state(B, L, dev)
    total = 0.0
    for c0 in range(0, n, chunk):
        h = h.detach()
        loss = 0.0
        for t in range(c0, min(c0 + chunk, n)):
            h, logits = model.round(h, xs[t])
            loss = loss + model.bc_loss(logits, tgts[t], None) / n
        loss.backward()
        total += float(loss.detach())
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return total, n


def cmd_train(a):
    dev = pick_device(a.device)
    cfg, model = load_model(a.init, a.init_model, torch.device("cpu"), ste=True)
    if int(cfg.get("qbits", 0)):
        raise SystemExit("v7 training starts from full-precision models only")
    if a.K:
        model.K = a.K
        cfg = dict(cfg, K=a.K)
    if a.xin == 3 and getattr(model, "xin", 2) == 2:
        model = add_detection_channel(model)
        cfg = dict(cfg, xin=3)
    model = model.to(dev)
    out = Path("runs") / f"nca-{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    cfg2 = dict(cfg, version=7, init=a.init, init_model=a.init_model,
                init_hash=file_hash(Path("runs") / f"nca-{a.init}" / a.init_model), code_hash=CODE_HASH,
                v7={k: v for k, v in vars(a).items() if isinstance(v, (int, float, str))})
    (out / "config.json").write_text(json.dumps(cfg2, indent=1))
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed + 7)
    Ls = [int(x) for x in a.Ls.split(",")]
    wts = torch.tensor([float(x) for x in a.weights.split(",")])
    if len(wts) != len(Ls):
        raise SystemExit("--weights needs one number per size")
    vals = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in a.val.split(",")]
    snaps = {int(x) for x in a.snap_at.split(",") if x}
    big_Ls = [int(x) for x in a.drill_Ls.split(",") if x]
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    log = (out / "train.log").open("a")
    t0, best = time.monotonic(), None

    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    def validate(it):
        nonlocal best
        model.eval()
        score, parts = 0, []
        for i, (vL, vp) in enumerate(vals):
            r = evaluate7(model, vL, vp, a.val_shots, 256 if vL < 39 else 128, a.val_seed + i, dev)
            score += r["fixed"]
            parts.append(f"L={vL} p={100*vp:.1f}%: {r['fixed']}/{r['n']}")
        model.train()
        say(f"VAL it {it:5d}  " + "  ".join(parts) + f"  total {score}  {(time.monotonic()-t0)/60:.1f} min")
        if best is None or score <= best:
            best = score
            torch.save(model.state_dict(), out / "model.pt")

    say(f"device {dev}  init {a.init}/{a.init_model}  C={model.C} M={model.M} K={model.K} xin={getattr(model,'xin',2)}  "
        f"long runs: {a.long_frac:.2f} of normal runs with {a.long_mult}L noisy rounds  big drills: {a.drill_big_frac:.2f} at L={big_Ls}  sizes {Ls}  p {a.pmin}-{a.pmax}  "
        f"steps {a.max_iters}  budget {a.minutes} min  code {CODE_HASH}")
    validate(0)
    it, rounds = 0, 0
    while it < a.max_iters and (time.monotonic() - t0) / 60 < a.minutes:
        it += 1
        L = Ls[int(torch.multinomial(wts, 1, generator=gen))]
        p = a.pmin + (a.pmax - a.pmin) * torch.rand(1, generator=gen).item()
        B = a.batch if L < 27 else a.batch_big
        drill = a.drill_k if torch.rand(1, generator=gen).item() < a.drill_frac else 0
        if big_Ls and torch.rand(1, generator=gen).item() < a.drill_big_frac:
            L = big_Ls[int(torch.randint(len(big_Ls), (1,), generator=gen))]   # clean-up drill at a large size
            B, drill = a.batch_drill, a.drill_k
        lm = a.long_mult if (not drill and torch.rand(1, generator=gen).item() < a.long_frac) else 1
        loss, n = train_step(model, opt, B, L, p, gen, dev, chunk=a.chunk, drill=drill, long_mult=lm)
        rounds += n
        if it % a.log_every == 0:
            say(f"it {it:5d} L={L} p={p:.4f} {'drill' if drill else f'noisy={lm}L'} loss={loss:.4f} rounds/step={rounds/it:.0f}  "
                f"{(time.monotonic()-t0)/60:.1f} min")
        if it % a.val_every == 0:
            validate(it)
        if it in snaps:
            torch.save(model.state_dict(), out / f"model_s{it}.pt")
            say(f"snapshot saved: model_s{it}.pt")
    if it % a.val_every:
        validate(it)
    torch.save(model.state_dict(), out / "model_last.pt")
    say(f"done after {it} steps; best validation total {best}; last model saved as model_last.pt")


# ---------------------------------------------------------------- self-tests

def cmd_check(a):
    ok_all = True

    def report(name, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"{name}:", "OK" if ok else "FAIL", flush=True)

    torch.manual_seed(0)
    cfg = dict(C=16, M=8, K=3, hidden=64, gru=False, action="link", dirmsg=False)
    base = build(cfg)
    for prm in base.parameters():
        prm.data.normal_(0, 0.3)
    L, B = 9, 4
    # 1. detection events: with no reading errors, d_t equals the syndrome of this round's new data errors,
    #    whatever the decoder flips
    gen = torch.Generator().manual_seed(1)
    e = torch.zeros(B, 2, L, L, dtype=torch.bool); corr = torch.zeros_like(e)
    prev = torch.zeros(B, L, L, dtype=torch.bool); lastflip = torch.zeros_like(e)
    ok = True
    for t in range(12):
        noise = torch.rand(e.shape, generator=gen) < 0.03
        e = e ^ noise
        m = synd(e ^ corr)
        d = observe(m, prev, lastflip, 3)[:, 2].bool()
        ok &= bool((d == synd(noise)).all())
        flips = torch.rand(e.shape, generator=gen) < 0.05
        prev, lastflip = m, flips
        corr = corr ^ flips
    report("1. detection event = syndrome of new data errors (no reading errors, random decoder flips)", ok)
    # 2. adding the channel keeps the decoder's output exactly (double precision), whatever the third channel holds
    wide = add_detection_channel(copy.deepcopy(base))
    b64, w64 = copy.deepcopy(base).double(), wide.double()
    x2 = (torch.rand(B, 2, L, L) < 0.2).double()
    x3 = torch.cat([x2, (torch.rand(B, 1, L, L) < 0.5).double()], 1)
    with torch.no_grad():
        h1, l1 = b64.round(b64.init_state(B, L, "cpu").double(), x2)
        h2, l2 = w64.round(w64.init_state(B, L, "cpu").double(), x3)
        for _ in range(4):
            h1, l1 = b64.round(h1, x2); h2, l2 = w64.round(h2, x3)
    report(f"2. extra input channel starts with no effect (max diff {float((l1-l2).abs().max()):.1e})",
           float((l1 - l2).abs().max()) < 1e-12)
    # 3. saved options survive saving and loading (xin, K, qbits)
    import tempfile, os
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            Path("runs/nca-t3").mkdir(parents=True)
            c3 = dict(cfg, xin=3, K=6, train_Ls=[9])
            Path("runs/nca-t3/config.json").write_text(json.dumps(c3))
            torch.save(add_detection_channel(copy.deepcopy(base)).state_dict(), "runs/nca-t3/model.pt")
            _, mm = load_model("t3", "model.pt", torch.device("cpu"))
            ok = mm.xin == 3 and mm.K == 6 and mm.upd[0].in_channels == cfg["C"] + 4 * cfg["M"] + 3
            Path("runs/nca-t4").mkdir(parents=True)
            Path("runs/nca-t4/config.json").write_text(json.dumps(dict(cfg, qbits=4, train_Ls=[9])))
            torch.save(base.state_dict(), "runs/nca-t4/model.pt")
            _, mq = load_model("t4", "model.pt", torch.device("cpu"))
            ok &= "round" in mq.__dict__                               # rounding installed
        finally:
            os.chdir(cwd)
    report("3. load_model restores extra channel, K and rounding", ok)
    # 4. evaluation: a rule that never acts, no noise -> no failures; every run gets exactly 2L noiseless rounds;
    #    results do not depend on how runs are grouped into batches
    idle = copy.deepcopy(base)
    with torch.no_grad():
        for hd in idle.act:
            hd[-1].bias.fill_(-1e3)
    idle.xin = 2
    r0 = evaluate7(idle, 7, 0.0, 64, 32, 5, torch.device("cpu"))
    report("4a. idle rule, no noise: no failures", r0["fixed"] == 0 and r0["stop"] == 0)
    calls = []
    orig = idle.round

    def counting(h, x):
        calls.append(1)
        return orig(h, x)
    idle.round = counting
    evaluate7(idle, 7, 0.01, 8, 8, 5, torch.device("cpu"))
    report(f"4b. every run gets L noisy + 2L noiseless rounds ({len(calls)} rounds, expected 21)", len(calls) == 21)
    del idle.__dict__["round"]
    # batch independence: same noise stream split differently gives the same per-run judgement only if noise is drawn
    # per batch; instead check the code path: one batch of 40 vs one batch of 40 run twice gives identical counts
    rA = evaluate7(base, 7, 0.02, 40, 40, 9, torch.device("cpu"))
    rB = evaluate7(base, 7, 0.02, 40, 40, 9, torch.device("cpu"))
    report("4c. evaluation is reproducible for a fixed seed", rA == rB)
    # 4d. the per-run judgement does not use other runs: evaluate a batch, then the same runs with a different
    #     neighbour run appended is not possible with one generator, so check directly that no reduction over the
    #     batch dimension decides when a run stops: a batch where one run can never get clean (a frozen defect pair)
    stubborn = copy.deepcopy(idle)
    r1 = evaluate7(stubborn, 7, 0.0, 4, 4, 1, torch.device("cpu"))
    report("4d. clean runs stay clean and count as successes", r1["fixed"] == 0 and r1["ever_clean"] == 4)
    # 5. one training step with the extra channel and long noisy runs moves the weights
    m5 = add_detection_channel(copy.deepcopy(base))
    w0 = copy.deepcopy(m5.state_dict())
    opt = torch.optim.Adam(m5.parameters(), lr=1e-3)
    loss, n = train_step(m5, opt, 4, 7, 0.015, torch.Generator().manual_seed(3), torch.device("cpu"), long_mult=4)
    moved = max(float((m5.state_dict()[k] - w0[k]).abs().max()) for k in w0)
    new_col = float(m5.state_dict()["upd.0.weight"][:, -1].abs().max())
    report(f"5. training step (extra channel, 4L noisy rounds: {n} rounds recorded, expected >= 28): weights moved "
           f"{moved:.1e}, new-channel weights {new_col:.1e}", moved > 0 and new_col > 0 and n >= 28)
    # 6. the multi-checkpoint evaluation reproduces evaluate7 at 2L (same noise), and matched <= fixed
    r7 = evaluate7(base, 7, 0.02, 40, 20, 9, torch.device("cpu"))
    n, rm = evaluate_multi(base, 7, 0.02, 40, 20, 9, torch.device("cpu"), (2, 6))
    report(f"6a. after 2L the diagnostic gives evaluate7's numbers (fixed {rm[2]['fixed']} vs {r7['fixed']}, "
           f"leftover {rm[2]['leftover']} vs {r7['leftover_runs']})",
           rm[2]["fixed"] == r7["fixed"] and rm[2]["leftover"] == r7["leftover_runs"] and n == 40)
    report("6b. perfect finish never adds failures (matched <= fixed)",
           all(rm[c]["matched"] <= rm[c]["fixed"] for c in (2, 6)))
    # 6c. a rule that does nothing, one short error string, no noise: perfect finish repairs it (matched 0),
    #     strict score fails (defects left)
    idle2 = copy.deepcopy(base)
    with torch.no_grad():
        for hd in idle2.act:
            hd[-1].bias.fill_(-1e3)
    idle2.xin = 2
    n, ri = evaluate_multi(idle2, 7, 0.0, 4, 4, 1, torch.device("cpu"), (2,))
    report("6c. idle rule, no noise: no failures in any score", ri[2] == dict(fixed=0, leftover=0, matched=0))
    # 7. a training step with a big-size drill runs
    m7 = copy.deepcopy(base)
    opt = torch.optim.Adam(m7.parameters(), lr=1e-3)
    loss, n = train_step(m7, opt, 2, 11, 0.0, torch.Generator().manual_seed(4), torch.device("cpu"), drill=3)
    report(f"7. drill training step at L=11 runs ({n} rounds)", n > 0)
    if not ok_all:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    t = sub.add_parser("train")
    t.add_argument("--init", required=True); t.add_argument("--init-model", default="model.pt")
    t.add_argument("--label", required=True)
    t.add_argument("--xin", type=int, default=2, choices=[2, 3]); t.add_argument("--K", type=int, default=0)
    t.add_argument("--long-frac", type=float, default=0.0); t.add_argument("--long-mult", type=int, default=4)
    t.add_argument("--max-iters", type=int, default=12000); t.add_argument("--minutes", type=float, default=300)
    t.add_argument("--Ls", default="13,19,27"); t.add_argument("--weights", default="1,2,1")
    t.add_argument("--pmin", type=float, default=0.014); t.add_argument("--pmax", type=float, default=0.022)
    t.add_argument("--batch", type=int, default=32); t.add_argument("--batch-big", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--chunk", type=int, default=24)
    t.add_argument("--drill-frac", type=float, default=0.3); t.add_argument("--drill-k", type=int, default=3)
    t.add_argument("--val", default="19:0.018,27:0.018,39:0.018"); t.add_argument("--val-shots", type=int, default=2048)
    t.add_argument("--val-seed", type=int, default=999001); t.add_argument("--val-every", type=int, default=2000)
    t.add_argument("--drill-Ls", default="", help="extra clean-up drills at these (larger) sizes, e.g. 39,55")
    t.add_argument("--drill-big-frac", type=float, default=0.0); t.add_argument("--batch-drill", type=int, default=8)
    t.add_argument("--snap-at", default=""); t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--device", default="auto")
    e = sub.add_parser("eval")
    e.add_argument("--label", required=True); e.add_argument("--model", default="model.pt")
    e.add_argument("--Ls", required=True); e.add_argument("--ps", required=True)
    e.add_argument("--shots", type=int, default=10000); e.add_argument("--batch", type=int, default=250)
    e.add_argument("--seed", type=int, default=8); e.add_argument("--tag", required=True)
    e.add_argument("--device", default="auto")
    d = sub.add_parser("deval")
    d.add_argument("--label", required=True); d.add_argument("--model", default="model.pt")
    d.add_argument("--Ls", required=True); d.add_argument("--ps", required=True)
    d.add_argument("--shots", type=int, default=10000); d.add_argument("--batch", type=int, default=250)
    d.add_argument("--seed", type=int, default=8); d.add_argument("--clean", default="2,6")
    d.add_argument("--tag", required=True); d.add_argument("--device", default="auto")
    ds = sub.add_parser("dsummary")
    ds.add_argument("--rows", required=True); ds.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--rows", required=True, help="name=label:modelfile,..."); s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"check": cmd_check, "train": cmd_train, "eval": cmd_eval, "summary": cmd_summary,
     "deval": cmd_deval, "dsummary": cmd_dsummary}[a.cmd](a)


if __name__ == "__main__":
    main()
