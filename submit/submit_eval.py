"""Big tests on SubMIT, split into many small independent pieces ("shards") that run as one Slurm job array.

Each shard = a fixed number of runs of one model at one (L, p), with its own seed, scored exactly like the Macs'
`decoder_v7.py deval --clean 2` (L noisy rounds, then 2L noiseless rounds; strict 'fixed', 'leftover', and
'matched' = logical error after a perfect matching finishes what is left). Each shard writes ONE small json file
(written to a temporary name, then renamed, so a killed job never leaves a half-written result). A finished shard is
never run twice. merge adds the shards up and refuses to mix shards from different models, code or settings.

  python submit_eval.py plan  --label v10-big --model model_last.pt --points 79:0.016:50000,111:0.017:30000 \
                              --shard 79:2000,111:800 --tag sub1 --tasks tasks-sub1.txt
  python submit_eval.py run   --tasks tasks-sub1.txt --index $SLURM_ARRAY_TASK_ID --out RESULTS_DIR
  python submit_eval.py merge --tasks tasks-sub1.txt --out RESULTS_DIR
Seeds: sha256 of (label, model, L, p, tag, shard) -> no two shards share noise, and none reuse the Macs' seeds.

Scoring (evaluate_long): every run gets L noisy rounds and 6L noiseless rounds and is scored after 2L, 4L and 6L
(identical to decoder_v7.evaluate_multi with cleans 2,4,6: same loop, same noise). Runs that still have defects after
6L continue alone (the noiseless phase draws no random numbers, so leaving out the clean runs changes nothing for the
others) and are scored again after 10L and 20L. Runs clean at 6L keep their 6L result for 10L and 20L; that a clean
run stays clean is checked on the full batch between 2L and 6L ('reopened_2to6': runs clean at 2L with defects at 6L).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent           # .../qec (decoder files, runs/)
EVAL_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
FIELDS = ("label", "model", "L", "p", "shots", "batch", "seed", "tag", "shard")


def shard_seed(label, model, L, p, tag, shard):
    s = f"{label}|{model}|{L}|{p:.6f}|{tag}|{shard}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 2**31


FULL, TAIL = (2, 4, 6), (10, 20)
PROTOCOL = "L noisy + noiseless; scored after 2,4,6 L (all runs) and 10,20 L (runs unfinished at 6L continue)"


def evaluate_long(model, L, p, shots, batch, seed, dev, full=FULL, tail=TAIL):
    import torch
    from decoder_v41 import rnd, synd, winding
    from decoder_v7 import observe
    from decoder_tests import ideal_fix
    gen = torch.Generator().manual_seed(seed)
    xin = getattr(model, "xin", 2)
    cs = list(full) + list(tail)
    out = {c: dict(fixed=0, leftover=0, matched=0) for c in cs}
    out["reopened_2to6"] = 0
    marks = {L + c * L: c for c in full}
    tmarks = {L + c * L: c for c in tail}
    n = 0
    with torch.no_grad():
        while n < shots:
            B = min(batch, shots - n)
            e = torch.zeros(B, 2, L, L, dtype=torch.bool, device=dev)
            corr = torch.zeros_like(e)
            prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
            lastflip = torch.zeros_like(e)
            h = model.init_state(B, L, dev)
            clean2 = None
            for t in range(L + max(full) * L):                    # exactly decoder_v7.evaluate_multi's loop
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
                    w = winding(d)
                    out[c]["fixed"] += int((w | left).sum())
                    out[c]["leftover"] += int(left.sum())
                    out[c]["matched"] += int(winding(d ^ ideal_fix(d)).sum())
                    if c == min(full):
                        clean2 = ~left
                    if c == max(full):
                        out["reopened_2to6"] += int((clean2 & left).sum())
                        w6 = w                                       # final 6L state of the clean runs
            d = e ^ corr
            left = synd(d).flatten(1).any(1)
            for c in tail:                                       # clean runs keep their 6L result
                out[c]["fixed"] += int((w6 & ~left).sum())
                out[c]["matched"] += int((w6 & ~left).sum())
            if left.any():                                       # only unfinished runs continue
                k = left
                e, corr, prev, lastflip, h = e[k], corr[k], prev[k], lastflip[k], h[k]
                for t in range(L + max(full) * L, L + max(tail) * L):
                    td = synd(e ^ corr)
                    h, logits = model.round(h, observe(td, prev, lastflip, xin))
                    flips = model.flips(model.greedy(logits))
                    prev, lastflip = td, flips
                    corr = corr ^ flips
                    if t + 1 in tmarks:
                        c = tmarks[t + 1]
                        d = e ^ corr
                        lf = synd(d).flatten(1).any(1)
                        out[c]["fixed"] += int((winding(d) | lf).sum())
                        out[c]["leftover"] += int(lf.sum())
                        out[c]["matched"] += int(winding(d ^ ideal_fix(d)).sum())
            n += B
    return n, out


def batch_for(L, shots, spec=None):
    if spec and L in spec:
        return min(shots, spec[L])
    return min(shots, 50 if L <= 79 else 32)


def read_tasks(path):
    tasks = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            v = line.split()
            t = dict(zip(FIELDS, v))
            for k in ("L", "shots", "batch", "seed", "shard"):
                t[k] = int(t[k])
            t["p"] = float(t["p"])
            tasks.append(t)
    return tasks


def shard_path(out, t):
    return Path(out) / t["label"] / t["tag"] / f"L{t['L']}_p{t['p']:.4f}" / f"shard{t['shard']:04d}.json"


def cmd_plan(a):
    per = {int(k): int(v) for k, v in (x.split(":") for x in a.shard.split(","))}
    bspec = {int(k): int(v) for k, v in (x.split(":") for x in a.batch.split(","))} if a.batch else None
    lines = ["# " + " ".join(FIELDS)]
    n_tasks = 0
    for spec in a.points.split(","):
        L, p, total = spec.split(":")
        L, p, total = int(L), float(p), int(total)
        if L not in per:
            raise SystemExit(f"no shard size given for L={L}")
        k = per[L]
        if total % k:
            raise SystemExit(f"L={L}: total {total} is not a multiple of the shard size {k}")
        for s in range(total // k):
            lines.append(" ".join(map(str, (a.label, a.model, L, f"{p:.4f}", k, batch_for(L, k, bspec),
                                            shard_seed(a.label, a.model, L, p, a.tag, s), a.tag, s))))
            n_tasks += 1
    seeds = [ln.split()[6] for ln in lines[1:]]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("seed collision; change --tag")
    Path(a.tasks).write_text("\n".join(lines) + "\n")
    print(f"{n_tasks} shards written to {a.tasks}. Submit with --array=0-{n_tasks - 1}%<max running>")


def cmd_run(a):
    tasks = read_tasks(a.tasks)
    todo = tasks[a.index * a.per_task:(a.index + 1) * a.per_task]
    if not todo:
        raise SystemExit(f"index {a.index}: no tasks (there are {len(tasks)})")
    sys.path.insert(0, str(CODE))
    os.chdir(CODE)
    import torch
    from decoder_v7 import CODE_HASH, file_hash, load_model
    dev = torch.device(a.device)
    models = {}
    for t in todo:
        f = shard_path(a.out, t)
        if f.exists():
            print(f"shard {f} already done", flush=True)
            continue
        key = (t["label"], t["model"])
        if key not in models:
            cfg, m = load_model(t["label"], t["model"], dev)
            m.eval()
            src = Path("runs") / f"nca-{t['label']}"
            models[key] = (m, dict(model_hash=file_hash(src / t["model"]),
                                   config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                                   K=m.K, C=m.C, M=m.M, xin=getattr(m, "xin", 2), qbits=int(cfg.get("qbits", 0))))
        m, prov = models[key]
        t0 = time.monotonic()
        n, res = evaluate_long(m, t["L"], t["p"], t["shots"], t["batch"], t["seed"], dev)
        row = dict(t, **prov, code_hash=CODE_HASH, eval_hash=EVAL_HASH, device=str(dev), host=os.uname().nodename,
                   job=os.environ.get("SLURM_JOB_ID"), n=n, reopened_2to6=res["reopened_2to6"],
                   seconds=round(time.monotonic() - t0, 1), protocol=PROTOCOL)
        for c in FULL + TAIL:
            for k in ("fixed", "leftover", "matched"):
                row[f"{k}_{c}L"] = res[c][k]
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(row))
        os.replace(tmp, f)
        print(f"{t['label']} L={t['L']} p={t['p']} shard {t['shard']} / {n} runs: " + "  ".join(
            f"{c}L fixed {row[f'fixed_{c}L']} left {row[f'leftover_{c}L']} matched {row[f'matched_{c}L']}"
            for c in FULL + TAIL) + f"  reopened {row['reopened_2to6']}  ({row['seconds']:.0f}s)", flush=True)


def cmd_checklong(a):
    """real model: evaluate_long must equal decoder_v7.evaluate_multi run in full to 20L (all runs) at every c."""
    sys.path.insert(0, str(CODE))
    os.chdir(CODE)
    import torch
    from decoder_v7 import evaluate_multi, load_model
    dev = torch.device(a.device)
    cfg, m = load_model(a.label, a.model, dev)
    m.eval()
    ok = True
    for L, p, shots, batch in ((27, 0.018, 2000, 250), (55, 0.017, 1000, 250), (79, 0.017, 500, 64)):
        t0 = time.monotonic()
        n1, x = evaluate_long(m, L, p, shots, batch, 4242 + L, dev)
        n2, y = evaluate_multi(m, L, p, shots, batch, 4242 + L, dev, cleans=FULL + TAIL)
        same = all(x[c] == y[c] for c in FULL + TAIL)
        ok &= same
        print(f"L={L} p={p} {shots} runs: " + "  ".join(f"{c}L {x[c]['fixed']}/{x[c]['leftover']}/{x[c]['matched']}"
              for c in FULL + TAIL) + f"  reopened {x['reopened_2to6']}  -> {'EQUAL' if same else 'DIFFERENT'}"
              f"  ({time.monotonic()-t0:.0f}s)", flush=True)
        if not same:
            print("   full-length reference:", {c: y[c] for c in FULL + TAIL}, flush=True)
    print("CHECKLONG OK" if ok else "CHECKLONG FAILED", flush=True)
    if not ok:
        raise SystemExit(1)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * (ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [max(0.0, c - h), min(1.0, c + h)]


def cmd_merge(a):
    tasks = read_tasks(a.tasks)
    groups, missing, bad = {}, [], []
    for t in tasks:
        f = shard_path(a.out, t)
        if not f.exists():
            missing.append(t)
            continue
        r = json.loads(f.read_text())
        for k in FIELDS:                                   # the file must belong to exactly this task
            if r.get(k) != t[k] and not (k == "p" and abs(r[k] - t[k]) < 1e-12):
                bad.append(f"{f}: {k} {r.get(k)} != {t[k]}")
        if r["n"] != t["shots"]:
            bad.append(f"{f}: n {r['n']} != {t['shots']}")
        groups.setdefault((t["label"], t["model"], t["L"], t["p"]), []).append(r)
    for g, rs in groups.items():
        for k in ("model_hash", "config_hash", "code_hash", "eval_hash", "protocol"):
            if len({r[k] for r in rs}) != 1:
                bad.append(f"{g}: shards disagree on {k}: {sorted({r[k] for r in rs})}")
        if len({r["seed"] for r in rs}) != len(rs):
            bad.append(f"{g}: repeated seeds")
    if bad:
        print("MERGE REFUSED:\n  " + "\n  ".join(bad[:50]))
        raise SystemExit(1)
    rows = []
    for (label, model, L, p), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3])):
        n = sum(r["n"] for r in rs)
        row = dict(label=label, model=model, L=L, p=p, shots=n, shards=len(rs),
                   shards_planned=sum(1 for t in tasks if (t["label"], t["model"], t["L"], t["p"]) == (label, model, L, p)),
                   model_hash=rs[0]["model_hash"], config_hash=rs[0]["config_hash"], code_hash=rs[0]["code_hash"],
                   eval_hash=rs[0]["eval_hash"],
                   protocol=rs[0]["protocol"], tag=rs[0]["tag"],
                   devices=sorted({r["device"] for r in rs}))
        sums = sorted(k for k in rs[0] if k.split("_")[0] in ("fixed", "leftover", "matched") or k == "reopened_2to6")
        for k in sums:
            row[k] = sum(r[k] for r in rs)
            if k.split("_")[0] in ("fixed", "matched"):
                row[k + "_rate"] = row[k] / n
                row[k + "_ci"] = wilson(row[k], n)
        row["cpu_hours"] = round(sum(r["seconds"] for r in rs) / 3600, 1)
        rows.append(row)
    tag = tasks[0]["tag"]
    out = Path(a.out) / f"merged-{tag}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lines = [f"# SubMIT tests, tag {tag} (SIMULATED). Failures / runs after c*L noiseless rounds.", "",
             "strict = logical error or defects left; left = runs with defects left; matched = logical error after a "
             "perfect matching finishes what is left (protection).", "",
             "| model | L | p | runs (shards) | strict 2L | strict 6L | strict 20L | left 2L | left 6L | left 20L | "
             "matched 2L | matched 20L | reopened 2->6L |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda k: f"{r[k]} ({r[k] / r['shots']:.1e})"
        lines.append(f"| {r['label']} [{r['model']}] | {r['L']} | {100*r['p']:.2f}% | {r['shots']} "
                     f"({r['shards']}/{r['shards_planned']}) | {f('fixed_2L')} | {f('fixed_6L')} | {f('fixed_20L')} | "
                     f"{r['leftover_2L']} | {r['leftover_6L']} | {r['leftover_20L']} | {f('matched_2L')} | "
                     f"{f('matched_20L')} | {r['reopened_2to6']} |")
    lines += ["", f"missing shards: {len(missing)} of {len(tasks)}"]
    (Path(a.out) / f"merged-{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--label", required=True); p.add_argument("--model", default="model_last.pt")
    p.add_argument("--points", required=True, help="L:p:total_runs,...")
    p.add_argument("--shard", required=True, help="L:runs_per_shard,...")
    p.add_argument("--batch", default="", help="L:batch,... (default 50 up to L79, else 32)")
    p.add_argument("--tag", required=True); p.add_argument("--tasks", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tasks", required=True); r.add_argument("--index", type=int, required=True)
    r.add_argument("--per-task", type=int, default=1); r.add_argument("--out", required=True)
    r.add_argument("--device", default="cpu")
    c = sub.add_parser("checklong")
    c.add_argument("--label", default="v10-big"); c.add_argument("--model", default="model_last.pt")
    c.add_argument("--device", default="cuda")
    m = sub.add_parser("merge")
    m.add_argument("--tasks", required=True); m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "merge": cmd_merge, "checklong": cmd_checklong}[a.cmd](a)


if __name__ == "__main__":
    main()
