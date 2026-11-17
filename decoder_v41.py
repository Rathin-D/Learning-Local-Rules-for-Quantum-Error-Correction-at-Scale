"""Version 4.1 of the learned fully local decoder (self-contained file). The RULE is unchanged from version 4.

Version 4 (trained on L = 9, 13, 19) beat or matched Lake at those sizes, but at the unseen L = 27 its failure rose.
The logs show the rise is mostly logical errors, and most leftover runs were still moving (not stuck) when the run
ended. Version 4.1 changes only what that result calls for, and nothing in the rule:
1. Exact teacher at every size. The old teacher paired defects exactly only up to 40 defects and fell back to a
   cruder greedy pairing above that. At L = 19 this never happened; at L = 27 it would for p above about 1.3%.
   Now exact minimum-weight pairing through PyMatching (also much faster). Needs: python -m pip install pymatching
2. Training includes L = 27 (test on L = 39, never seen), and is longer (version 4 was still improving at the end).
3. Checkpoint chosen at the largest size with 2048 fixed runs (version 4's last 1000 steps all scored 0-5 failures
   of 512, so its choice was within noise).
4. Evaluation uses 10,000 runs per point, so a fall or rise with size can be told apart from noise.
5. The size weights must match the list of sizes (the old code would silently never train the last size).
Benchmark unchanged (Lake arXiv:2510.08056: toric code, one sector, q = p, L noisy rounds, then up to 2L noiseless
rounds; strict failure = logical error or leftover defects). K = 3 message ticks per round, 16 memory values and
8 message values per site.

Usage (Mac: conda base python, GPU used automatically):
  python decoder_v41.py check
  python decoder_v41.py train --label v41 --Ls 9,13,19,27 --weights 1,1,2,3 --bc-iters 8000
  python decoder_v41.py eval --label v41 --Ls 9,13,19,27,39 --shots 10000
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lake's decoder in the same scenario (logical failure per run, 3000 runs per point); his reported threshold ~1.5%.
LAKE_REF = {
    9: {0.0075: 0.0023, 0.01: 0.0097, 0.012: 0.0160, 0.015: 0.0453, 0.018: 0.0980, 0.022: 0.2067},
    13: {0.0075: 0.0017, 0.01: 0.0030, 0.012: 0.0117, 0.015: 0.0417, 0.018: 0.1237, 0.022: 0.3180},
    19: {0.0075: 0.0000, 0.01: 0.0017, 0.012: 0.0063, 0.015: 0.0400, 0.018: 0.1600, 0.022: 0.4910},
}
BIG = 1.0e6

# ---------------------------------------------------------------- devices and random numbers

def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rnd(shape, gen, dev):
    return torch.rand(shape, generator=gen).to(dev)

# ---------------------------------------------------------------- toric code (bit-flip sector)
# link e[:, 0, i, j] joins sites (i, j)-(i+1, j); e[:, 1, i, j] joins (i, j)-(i, j+1)
# moves: 0 stay, 1 +x (i+1), 2 +y (j+1), 3 -x, 4 -y

def synd(e):
    return e[:, 0] ^ e[:, 1] ^ torch.roll(e[:, 0], 1, 1) ^ torch.roll(e[:, 1], 1, 2)


def winding(e):
    return (e[:, 1, :, 0].sum(1) % 2 == 1) | (e[:, 0, 0, :].sum(1) % 2 == 1)


def shift_to(t, k):
    """value at the neighbour in direction k (0 +x, 1 +y, 2 -x, 3 -y); lattice dims are the last two."""
    return {0: torch.roll(t, -1, -2), 1: torch.roll(t, -1, -1), 2: torch.roll(t, 1, -2), 3: torch.roll(t, 1, -1)}[k]


def apply_moves(a):
    """site moves (B, L, L) -> link flips (B, 2, L, L): a link flips if the site at either end steps across it."""
    c0 = (a == 1) | shift_to(a == 3, 0)
    c1 = (a == 2) | shift_to(a == 4, 1)
    return torch.stack([c0, c1], 1)

# ---------------------------------------------------------------- teachers (see the true defects; training only)

_TAB = {}


def tables(L, dev):
    key = (L, str(dev))
    if key not in _TAB:
        ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        ci, cj = ii.flatten(), jj.flatten()
        dx = (ci[None, :] - ci[:, None]) % L
        dx = torch.where(dx > L // 2, dx - L, dx)
        dy = (cj[None, :] - cj[:, None]) % L
        dy = torch.where(dy > L // 2, dy - L, dy)
        dist = (dx.abs() + dy.abs()).float()
        dist.fill_diagonal_(BIG)
        one = torch.ones_like(dx)
        # partner in +x: step +x; otherwise step along y if dy != 0; otherwise step -x  (partners always converge)
        dirn = torch.where(dx > 0, one, torch.where(dy > 0, 2 * one, torch.where(dy < 0, 4 * one, 3 * one)))
        _TAB[key] = (dist.to(dev), dirn.to(dev))
    return _TAB[key]


def teacher_nearest(defect):
    """each true defect steps towards the nearest other true defect. (B, L, L) bool -> (B, L, L) long."""
    B, L, _ = defect.shape
    N = L * L
    dist, dirn = tables(L, defect.device)
    df = defect.flatten(1)
    d = dist[None].expand(B, N, N).masked_fill(~df[:, None, :], BIG)
    mn, arg = d.min(2)
    rows = torch.arange(N, device=defect.device)[None, :].expand(B, N)
    tgt = dirn[rows, arg]
    return torch.where(df & (mn < BIG), tgt, torch.zeros_like(tgt)).view(B, L, L)


def _tdist(a, b, L):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return min(dx, L - dx) + min(dy, L - dy)


def _pairs(pts, L):
    import networkx as nx
    n = len(pts)
    if n <= 40:
        G = nx.Graph()
        for u in range(n):
            for v in range(u + 1, n):
                G.add_edge(u, v, weight=-_tdist(pts[u], pts[v], L))
        return list(nx.max_weight_matching(G, maxcardinality=True))
    left, out = set(range(n)), []
    for d, u, v in sorted((_tdist(pts[u], pts[v], L), u, v) for u in range(n) for v in range(u + 1, n)):
        if u in left and v in left:
            out.append((u, v)); left -= {u, v}
    return out


_PM, _HAVE_PM = {}, None


def have_pymatching():
    global _HAVE_PM
    if _HAVE_PM is None:
        try:
            import pymatching                                        # noqa: F401
            _HAVE_PM = True
        except Exception:
            _HAVE_PM = False
    return _HAVE_PM


def _pm_graph(L):
    if L not in _PM:
        import pymatching
        m = pymatching.Matching()
        for i in range(L):
            for j in range(L):
                u = i * L + j
                m.add_edge(u, ((i + 1) % L) * L + j, weight=1.0)
                m.add_edge(u, i * L + (j + 1) % L, weight=1.0)
        _PM[L] = m
    return _PM[L]


def teacher_match(defect):
    """minimum-weight matching of the true defects (global; CPU); each defect steps towards its partner.
    Exact at any size with PyMatching; without it, exact up to 40 defects and greedy above."""
    dev = defect.device
    defect = defect.cpu()
    B, L, _ = defect.shape
    _, dirn = tables(L, torch.device("cpu"))
    if have_pymatching():
        m = _pm_graph(L)
        syn = defect.flatten(1).numpy().astype("uint8")
        flat = torch.zeros(B, L * L, dtype=torch.long)
        for b in range(B):
            if syn[b].any():
                pr = torch.from_numpy(m.decode_to_matched_dets_array(syn[b]).astype("int64"))
                flat[b, pr[:, 0]] = dirn[pr[:, 0], pr[:, 1]]
                flat[b, pr[:, 1]] = dirn[pr[:, 1], pr[:, 0]]
        return flat.view(B, L, L).to(dev)
    tgt = torch.zeros(B, L, L, dtype=torch.long)
    per = {}
    for b, i, j in defect.nonzero().tolist():
        per.setdefault(b, []).append((i, j))
    for b, pts in per.items():
        for u, v in _pairs(pts, L):
            for s, d in ((pts[u], pts[v]), (pts[v], pts[u])):
                tgt[b, s[0], s[1]] = dirn[s[0] * L + s[1], d[0] * L + d[1]]
    return tgt.to(dev)


TEACHERS = {"nearest": teacher_nearest, "match": teacher_match}

# ---------------------------------------------------------------- the local rule

class Rule(nn.Module):
    """One network, identical at every site. Per round: K ticks of nearest-neighbour messages (M numbers to each
    neighbour, different per direction), a C-number memory per site, then one move per site."""

    def __init__(self, C=16, M=8, K=3, hidden=64, gru=False, action="link", dirmsg=False):
        super().__init__()
        self.C, self.M, self.K, self.gru, self.action, self.dirmsg = C, M, K, gru, action, dirmsg
        self.msg = nn.Conv2d(C, (4 if dirmsg else 1) * M, 1)
        nin = C + 4 * M + 2
        if gru:
            self.gates = nn.Sequential(nn.Conv2d(nin, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 2 * C, 1))
            self.cand = nn.Sequential(nn.Conv2d(nin, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, C, 1))
        else:
            self.upd = nn.Sequential(nn.Conv2d(nin, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, C, 1))
        if action == "site":                                           # one move per site
            self.act = nn.Sequential(nn.Conv2d(C, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 5, 1))
            with torch.no_grad():
                self.act[-1].bias.copy_(torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]))  # start as "stay"
        elif action == "link":                                         # each link decided by its two end sites
            self.act = nn.ModuleList([nn.Sequential(nn.Conv2d(2 * C, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 1, 1))
                                      for _ in range(2)])
            for head in self.act:
                nn.init.constant_(head[-1].bias, -4.0)                 # start as "do nothing"
        else:
            raise ValueError(f"unknown action {action!r} (use link or site)")
        self.no_messages = False
        self.pos_weight = 1.0

    def init_state(self, B, L, dev):
        return torch.zeros(B, self.C, L, L, device=dev)

    def round(self, h, x):
        B, _, L, _ = h.shape
        for _ in range(self.K):
            m = torch.tanh(self.msg(h))
            if self.no_messages:
                m = torch.zeros_like(m)
            if self.dirmsg:                                               # m[:, k] is sent towards direction k
                m = m.view(B, 4, self.M, L, L)
                inc = [shift_to(m[:, (k + 2) % 4], k) for k in range(4)]  # from the neighbour in direction k
            else:                                                         # same message to all four neighbours
                inc = [shift_to(m, k) for k in range(4)]
            u = torch.cat([h, *inc, x], 1)
            if self.gru:
                zr = torch.sigmoid(self.gates(u))
                z, r = zr[:, :self.C], zr[:, self.C:]
                hc = torch.tanh(self.cand(torch.cat([r * h, *inc, x], 1)))
                h = (1 - z) * h + z * hc
            else:
                h = torch.tanh(h + self.upd(u))
        if self.action == "site":
            return h, self.act(h)
        l0 = self.act[0](torch.cat([h, shift_to(h, 0)], 1))[:, 0]
        l1 = self.act[1](torch.cat([h, shift_to(h, 1)], 1))[:, 0]
        return h, torch.stack([l0, l1], 1)

    # ---- acting and scoring, for both kinds of output (site moves: long (B,L,L); link flips: bool (B,2,L,L))
    def target(self, moves):
        return moves if self.action == "site" else apply_moves(moves)

    def greedy(self, logits):
        return logits.argmax(1) if self.action == "site" else logits > 0

    def sample(self, logits, gen, dev):
        u = rnd(logits.shape, gen, dev).clamp(1e-9, 1 - 1e-9)
        if self.action == "site":
            return (logits - torch.log(-torch.log(u))).argmax(1)
        return u < torch.sigmoid(logits)

    def flips(self, a):
        return apply_moves(a) if self.action == "site" else a

    def bc_loss(self, logits, tgt, active):
        if self.action == "site":
            ce = F.cross_entropy(logits, tgt, reduction="none")
            return ce.mean() + (ce * active).sum() / active.sum().clamp(min=1.0)
        pw = torch.tensor(self.pos_weight, device=logits.device) if self.pos_weight != 1.0 else None
        return F.binary_cross_entropy_with_logits(logits, tgt.float(), pos_weight=pw)

    def logp(self, logits, a):
        if self.action == "site":
            return -F.cross_entropy(logits, a, reduction="none").flatten(1).sum(1)
        a = a.float()
        return (a * F.logsigmoid(logits) + (1 - a) * F.logsigmoid(-logits)).flatten(1).sum(1)

# ---------------------------------------------------------------- training step (memory-bounded)

def drill_errors(B, L, k, gen):
    """clean-up drill: 1..k random error strings per run, so defect pairs start up to L/2 apart in x and in y."""
    e = torch.zeros(B, 2, L, L, dtype=torch.bool)
    r = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=gen))      # noqa: E731
    for b in range(B):
        for _ in range(r(1, k + 1)):
            i, j, dx, dy = r(0, L), r(0, L), r(-(L // 2), L // 2 + 1), r(-(L // 2), L // 2 + 1)
            for _ in range(abs(dx)):
                if dx > 0:
                    e[b, 0, i, j] ^= True; i = (i + 1) % L
                else:
                    i = (i - 1) % L; e[b, 0, i, j] ^= True
            for _ in range(abs(dy)):
                if dy > 0:
                    e[b, 1, i, j] ^= True; j = (j + 1) % L
                else:
                    j = (j - 1) % L; e[b, 1, i, j] ^= True
    return e


def train_step(model, opt, B, L, p, gen, mode, dev, teacher="match", chunk=24, beta=0.0, bc_coef=0.3, lam=0.2,
               drill=0):
    """Run one batch of episodes without gradients (recording inputs, teacher moves and the rule's own moves), then
    replay it with gradients in chunks of `chunk` rounds.
    mode 'bc': imitate the teacher; the rule acts with its own most likely move (with probability beta the teacher's).
    mode 'rl': moves are sampled; REINFORCE with reward = -(failure) - lam * (defects remaining, summed over the rest
    of the run), plus bc_coef x imitation loss.
    drill > 0: a clean-up drill instead of a normal run (no noise; up to `drill` far-apart defect pairs to bring
    together), which trains exactly the situation in which runs fail (defects left over at the end)."""
    tf = TEACHERS[teacher]
    T_noisy = 0 if drill else L
    e = drill_errors(B, L, drill, gen).to(dev) if drill else torch.zeros(B, 2, L, L, dtype=torch.bool, device=dev)
    corr = torch.zeros_like(e)
    prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
    h = model.init_state(B, L, dev)
    xs, tgts, acts, actives, ndefs = [], [], [], [], []
    with torch.no_grad():
        for t in range(T_noisy + 2 * L):
            noisy = t < T_noisy
            if noisy:
                e = e ^ (rnd(e.shape, gen, dev) < p)
            td = synd(e ^ corr)
            if not noisy and not bool(td.any()):
                break
            m = td ^ (rnd(td.shape, gen, dev) < p) if noisy else td
            x = torch.stack([m, m ^ prev], 1).float()
            prev = m
            tgt = model.target(tf(td))
            h, logits = model.round(h, x)
            if mode == "rl":
                a = model.sample(logits, gen, dev)
            else:
                a = model.greedy(logits)
                if beta > 0:
                    mix = rnd((B,) + (1,) * (tgt.dim() - 1), gen, dev) < beta
                    a = torch.where(mix, tgt, a)
            xs.append(x); tgts.append(tgt); acts.append(a)
            actives.append((td | m).float()); ndefs.append(td.flatten(1).sum(1).float())
            corr = corr ^ model.flips(a)
    d = e ^ corr
    sd = synd(d)
    fail = (winding(d) | sd.flatten(1).any(1)).float()
    n = len(xs)
    if n == 0:
        return 0.0, fail
    if mode == "rl":
        cost = torch.stack(ndefs[1:] + [sd.flatten(1).sum(1).float()], 0) / (L * L)       # (n, B)
        togo = torch.flip(torch.cumsum(torch.flip(cost, [0]), 0), [0])
        G = -fail[None, :] - lam * togo
        adv = (G - G.mean(1, keepdim=True)) / G.std().clamp(min=1e-3)
    opt.zero_grad()
    h = model.init_state(B, L, dev)
    total = 0.0
    for c0 in range(0, n, chunk):
        h = h.detach()
        loss = 0.0
        for t in range(c0, min(c0 + chunk, n)):
            h, logits = model.round(h, xs[t])
            nact = actives[t].sum().clamp(min=1.0)
            bc = model.bc_loss(logits, tgts[t], actives[t])
            if mode == "rl":
                lp = model.logp(logits, acts[t])
                loss = loss + (-(adv[t] * lp).mean() / (nact / B) + bc_coef * bc) / n
            else:
                loss = loss + bc / n
        loss.backward()
        total += float(loss.detach())
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return total, fail

# ---------------------------------------------------------------- evaluation (no teacher, no truth used to act)

def evaluate(model, L, p, shots, batch, seed, dev):
    gen = torch.Generator().manual_seed(seed)
    out = dict(native=0, strict=0, n=0, residual=0, leftover=0, stuck=0)
    with torch.no_grad():
        while out["n"] < shots:
            B = min(batch, shots - out["n"])
            e = torch.zeros(B, 2, L, L, dtype=torch.bool, device=dev)
            corr = torch.zeros_like(e)
            prev = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
            h = model.init_state(B, L, dev)
            last = None
            for t in range(3 * L):
                noisy = t < L
                if noisy:
                    e = e ^ (rnd(e.shape, gen, dev) < p)
                td = synd(e ^ corr)
                if not noisy and not bool(td.any()):
                    break
                m = td ^ (rnd(td.shape, gen, dev) < p) if noisy else td
                h, logits = model.round(h, torch.stack([m, m ^ prev], 1).float())
                prev, last = m, td
                corr = corr ^ model.flips(model.greedy(logits))
            d = e ^ corr
            sd = synd(d)
            w, r = winding(d), sd.flatten(1).any(1)
            out["native"] += int(w.sum()); out["strict"] += int((w | r).sum()); out["n"] += B
            out["residual"] += int(r.sum()); out["leftover"] += int(sd.flatten(1).sum(1)[r].sum())
            if last is not None:
                out["stuck"] += int(((sd == last).flatten(1).all(1) & r).sum())
    return out


def oracle(L, p, B, gen, teacher):
    """the teacher acting by itself (it sees the true defects): what its moves can achieve at best."""
    tf = TEACHERS[teacher]
    e = torch.zeros(B, 2, L, L, dtype=torch.bool)
    corr = torch.zeros_like(e)
    for t in range(3 * L):
        if t < L:
            e = e ^ (torch.rand(e.shape, generator=gen) < p)
        corr = corr ^ apply_moves(tf(synd(e ^ corr)))
    d = e ^ corr
    return float((winding(d) | synd(d).flatten(1).any(1)).float().mean())


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph, den = k / n, 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))

# ---------------------------------------------------------------- commands

def build(cfg):
    return Rule(cfg["C"], cfg["M"], cfg["K"], cfg["hidden"], cfg["gru"], cfg["action"], cfg["dirmsg"])


def cmd_train(a):
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path("runs") / f"nca-{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    cfg = dict(C=a.C, M=a.M, K=a.K, hidden=a.hidden, gru=a.gru, action=a.action, dirmsg=a.dir_msg,
               version=4, teacher=a.teacher, drill_frac=a.drill_frac, drill_k=a.drill_k, pos_weight=a.pos_weight,
               train_Ls=a.Ls,
               weights=a.weights, pmin=a.pmin, pmax=a.pmax, bc_iters=a.bc_iters, rl_iters=a.rl_iters, batch=a.batch,
               batch_big=a.batch_big, chunk=a.chunk, lr=a.lr, rl_lr=a.rl_lr, seed=a.seed)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    dev = pick_device(a.device)
    model = build(cfg).to(dev)
    model.pos_weight = a.pos_weight
    gen = torch.Generator().manual_seed(a.seed + 1)
    Ls = [int(x) for x in a.Ls.split(",")]
    wts = torch.tensor([float(x) for x in a.weights.split(",")])
    if len(wts) != len(Ls):
        raise SystemExit(f"--weights needs one number per size: got {len(wts)} weights for sizes {Ls}")
    vL = a.val_L or max(Ls)
    log = (out / "train.log").open("a")
    t0, best = time.monotonic(), None

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    say("teacher pairing: " + ("exact at every size (PyMatching)" if have_pymatching() else
        "exact up to 40 defects, GREEDY above (install with: python -m pip install pymatching)"))
    say(f"device: {dev}  teacher: {a.teacher}  sizes: {Ls}  gru: {cfg['gru']}  action: {a.action}  "
        f"direction-specific messages: {cfg['dirmsg']}")

    def pick(it=None):
        w = wts.clone()
        if it is not None and len(Ls) > 1 and it < a.warm * a.bc_iters:
            w[-1] = 0                                                   # curriculum: largest size joins later
        L = Ls[int(torch.multinomial(w, 1, generator=gen))]
        return L, a.pmin + (a.pmax - a.pmin) * torch.rand(1, generator=gen).item(), (a.batch if L < 19 else a.batch_big)

    def validate(it, phase):
        nonlocal best
        r = evaluate(model, vL, a.val_p, a.val_shots, 256, seed=424242, dev=dev)      # same noise every time
        score = (r["strict"] / r["n"], r["native"] / r["n"])
        say(f"{phase} it {it:5d}  val L={vL} p={a.val_p}: strict fail {score[0]:.4f} (logical {score[1]:.4f}, "
            f"leftover runs {r['residual']}, Lake {LAKE_REF.get(vL, {}).get(a.val_p, float('nan')):.4f})  "
            f"{time.monotonic()-t0:.0f}s")
        if best is None or score <= best:
            best = score
            torch.save(model.state_dict(), out / "model.pt")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(a.bc_iters, 1), eta_min=a.lr * 0.05)
    for it in range(1, a.bc_iters + 1):
        L, p, B = pick(it)
        beta = max(0.0, a.beta0 * (1 - it / max(1.0, a.beta_frac * a.bc_iters)))
        drill = a.drill_k if torch.rand(1, generator=gen).item() < a.drill_frac else 0
        train_step(model, opt, B, L, p, gen, "bc", dev, teacher=a.teacher, chunk=a.chunk, beta=beta, drill=drill)
        sched.step()
        if it % a.log_every == 0 or it == a.bc_iters:
            validate(it, "BC")
    torch.save(model.state_dict(), out / "model_bc.pt")
    if a.rl_iters > 0:
        model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
        opt = torch.optim.Adam(model.parameters(), lr=a.rl_lr)
        for it in range(1, a.rl_iters + 1):
            L, p, B = pick()
            train_step(model, opt, B, L, p, gen, "rl", dev, teacher=a.teacher, chunk=a.chunk, bc_coef=a.bc_coef,
                       lam=a.lam)
            if it % a.log_every == 0 or it == a.rl_iters:
                validate(a.bc_iters + it, "RL")
    torch.save(model.state_dict(), out / "model_last.pt")
    say(f"saved {out/'model.pt'} (best validation strict failure {best[0]:.4f})")


def cmd_eval(a):
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path("runs") / f"nca-{a.label}"
    cfg = json.loads((out / "config.json").read_text())
    dev = pick_device(a.device)
    model = build(cfg).to(dev)
    model.load_state_dict(torch.load(out / a.model, map_location=dev))
    model.no_messages = a.no_messages
    model.eval()
    tag = ("nomsg-" if a.no_messages else "") + a.tag
    jl = out / f"eval-{tag}.jsonl"
    rows = [json.loads(x) for x in jl.read_text().splitlines()] if jl.exists() else []
    done = {(r["L"], r["p"]) for r in rows}
    for L in map(int, a.Ls.split(",")):
        for p in map(float, a.ps.split(",")):
            if (L, p) in done:
                continue
            t0 = time.monotonic()
            r = evaluate(model, L, p, a.shots, a.batch, seed=(L * 1000003 + int(p * 1e6) + a.seed) % 2**31, dev=dev)
            n = r["n"]
            row = dict(L=L, p=p, shots=n, rate_strict=r["strict"] / n, ci_strict=wilson(r["strict"], n),
                       rate_native=r["native"] / n, residual=r["residual"], leftover=r["leftover"], stuck=r["stuck"],
                       lake=LAKE_REF.get(L, {}).get(p), seconds=time.monotonic() - t0)
            rows.append(row)
            with jl.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"L {L:3d} p {p:.4f}: strict {row['rate_strict']:.4f} logical {row['rate_native']:.4f} "
                  f"leftover runs {r['residual']} (stuck {r['stuck']})  Lake {row['lake']}  ({row['seconds']:.0f}s)",
                  flush=True)
    md = [f"# Learned local decoder `{a.label}`{' (messages removed)' if a.no_messages else ''} vs Lake (SIMULATED)",
          "", "Lake arXiv:2510.08056 scenario: toric code, one sector, q = p, L noisy rounds then up to 2L noiseless.",
          f"strict = logical error or leftover defects. Trained on L = {cfg['train_Ls']}; other sizes never seen.",
          "Lake column: his decoder, same protocol ('-' = not computed at this size).", "",
          "| L | p | shots | strict | 95% CI | logical only | runs with leftovers | Lake | strict / Lake |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["L"], r["p"])):
        ratio = f"{r['rate_strict']/r['lake']:.2f}" if r["lake"] else "-"
        md.append(f"| {r['L']} | {r['p']:.4f} | {r['shots']} | {r['rate_strict']:.4f} | {r['ci_strict'][0]:.4f}-"
                  f"{r['ci_strict'][1]:.4f} | {r['rate_native']:.4f} | {r['residual']} | "
                  f"{r['lake'] if r['lake'] is not None else '-'} | {ratio} |")
    (out / f"eval-{tag}.md").write_text("\n".join(md) + "\n")
    print(f"done: {out / f'eval-{tag}.md'}")


def cmd_check(a):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    cpu = torch.device("cpu")
    # 1. moves and teachers: a pair two sites apart annihilates in one round
    ok = True
    for name, tf in TEACHERS.items():
        e = torch.zeros(1, 2, 7, 7, dtype=torch.bool)
        e[0, 0, 1, 3] = e[0, 0, 2, 3] = True                          # defects at (1,3) and (3,3)
        left = int(synd(e ^ apply_moves(tf(synd(e)))).sum())
        e2 = torch.zeros(1, 2, 7, 7, dtype=torch.bool); e2[0, 0, 1, 1] = e2[0, 1, 2, 1] = True   # diagonal pair
        c = torch.zeros_like(e2)
        for _ in range(3):
            c = c ^ apply_moves(tf(synd(e2 ^ c)))
        left2 = int(synd(e2 ^ c).sum())
        ok &= left == 0 and left2 == 0
        print(f"teacher '{name}': straight pair left {left}, diagonal pair left {left2}", "ok" if left == 0 and left2 == 0 else "FAIL")
    # 2. what each teacher achieves acting alone (it sees the true defects)
    for name in TEACHERS:
        t0 = time.monotonic()
        res = [f"L={L} p={p}: {oracle(L, p, 200, gen, name):.3f} (Lake {LAKE_REF[L][p]:.4f})" for L, p in ((9, 0.01), (13, 0.015))]
        print(f"teacher '{name}' alone: " + "; ".join(res) + f"  ({time.monotonic()-t0:.1f}s)")
    # 2b. exact pairing at large sizes
    if have_pymatching():
        gq = torch.Generator().manual_seed(5)
        d9 = (torch.rand(40, 2, 9, 9, generator=gq) < 0.06)
        s9 = synd(d9)
        _, dirn9 = tables(9, cpu)
        dist9, _ = tables(9, cpu)
        wpm = wnx = 0.0
        for b in range(40):
            pts = [tuple(x) for x in s9[b].nonzero().tolist()]
            if not pts:
                continue
            pr = _pm_graph(9).decode_to_matched_dets_array(s9[b].flatten().numpy().astype("uint8"))
            wpm += sum(_tdist(divmod(int(u), 9), divmod(int(v), 9), 9) for u, v in pr)
            wnx += sum(_tdist(pts[u], pts[v], 9) for u, v in _pairs(pts, 9))
        s27 = synd(torch.rand(16, 2, 27, 27, generator=gq) < 0.02)
        t0 = time.monotonic()
        teacher_match(s27)
        print(f"exact teacher (PyMatching): total pairing length {wpm:.0f} vs reference {wnx:.0f}",
              "ok" if wpm == wnx else "FAIL", f"; 16 runs at L = 27 with {int(s27.sum())//16} defects each: "
              f"{(time.monotonic()-t0)*1000:.0f} ms")
    else:
        print("PyMatching NOT installed: teacher is greedy above 40 defects. Run: python -m pip install pymatching")
    # 3. locality
    model = Rule()
    x0 = torch.zeros(1, 2, 11, 11); x1 = x0.clone(); x1[0, :, 5, 5] = 1
    with torch.no_grad():
        ha, _ = model.round(model.init_state(1, 11, cpu), x0)
        hb, _ = model.round(model.init_state(1, 11, cpu), x1)
    idx = ((ha - hb).abs().sum(1)[0] > 1e-7).nonzero()
    reach = int((idx - 5).abs().sum(1).max()) if len(idx) else 0
    print(f"locality: one round spreads a change {reach} sites (must be <= {model.K})", "ok" if reach <= model.K else "FAIL")
    # 4. learning on every device
    devs = [cpu] + ([pick_device("auto")] if pick_device("auto").type != "cpu" else [])
    for dev in devs:
        torch.manual_seed(0)
        mdl = Rule().to(dev)
        opt = torch.optim.Adam(mdl.parameters(), lr=2e-3)
        t0, losses = time.monotonic(), []
        for i in range(30):
            l, _ = train_step(mdl, opt, 16, 7, 0.015, gen, "bc", dev, beta=0.5)
            losses.append(l)
        train_step(mdl, opt, 8, 7, 0.015, gen, "bc", dev, drill=3)
        print(f"[{dev.type}] imitation: loss {losses[0]:.3f} -> {sum(losses[-5:])/5:.3f} in 30 steps "
              f"({time.monotonic()-t0:.1f}s)", "ok" if sum(losses[-5:]) / 5 < losses[0] else "CHECK")
        train_step(mdl, opt, 8, 7, 0.015, gen, "rl", dev)
        train_step(mdl, opt, 4, 7, 0.015, gen, "bc", dev, teacher="nearest")
        print(f"[{dev.type}] drill, RL and nearest-teacher steps: ok")
        for L, B in ((19, 16), (27, 16)):
            t0 = time.monotonic()
            train_step(mdl, opt, B, L, 0.012, gen, "bc", dev)
            print(f"[{dev.type}] one training step at L = {L}, batch {B}: {time.monotonic()-t0:.1f}s")
        t0 = time.monotonic()
        evaluate(mdl, 13, 0.012, 64, 64, seed=3, dev=dev)
        print(f"[{dev.type}] evaluation of 64 runs at L = 13: {time.monotonic()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    t = sub.add_parser("train")
    t.add_argument("--label", default="v41"); t.add_argument("--Ls", default="9,13,19,27")
    t.add_argument("--weights", default="1,1,2,3"); t.add_argument("--warm", type=float, default=0.2)
    t.add_argument("--teacher", default="match", choices=list(TEACHERS))
    t.add_argument("--pmin", type=float, default=0.005); t.add_argument("--pmax", type=float, default=0.02)
    t.add_argument("--bc-iters", type=int, default=8000); t.add_argument("--rl-iters", type=int, default=0)
    t.add_argument("--batch", type=int, default=32); t.add_argument("--batch-big", type=int, default=16)
    t.add_argument("--chunk", type=int, default=24); t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--rl-lr", type=float, default=1e-4); t.add_argument("--bc-coef", type=float, default=0.3)
    t.add_argument("--lam", type=float, default=0.2)
    t.add_argument("--beta0", type=float, default=0.5); t.add_argument("--beta-frac", type=float, default=0.3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--C", type=int, default=16); t.add_argument("--M", type=int, default=8)
    t.add_argument("--K", type=int, default=3); t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--gru", action="store_true"); t.add_argument("--dir-msg", action="store_true")
    t.add_argument("--action", default="link", choices=["link", "site"])
    t.add_argument("--drill-frac", type=float, default=0.3); t.add_argument("--drill-k", type=int, default=3)
    t.add_argument("--pos-weight", type=float, default=1.0)
    t.add_argument("--log-every", type=int, default=200); t.add_argument("--val-p", type=float, default=0.012)
    t.add_argument("--val-L", type=int, default=0); t.add_argument("--val-shots", type=int, default=2048)
    t.add_argument("--device", default="auto"); t.add_argument("--threads", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--label", default="v41"); e.add_argument("--Ls", default="9,13,19,27,39")
    e.add_argument("--ps", default="0.0075,0.01,0.012,0.015,0.018"); e.add_argument("--shots", type=int, default=10000)
    e.add_argument("--batch", type=int, default=250); e.add_argument("--seed", type=int, default=7)
    e.add_argument("--no-messages", action="store_true"); e.add_argument("--tag", default="main")
    e.add_argument("--model", default="model.pt"); e.add_argument("--device", default="auto")
    e.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    {"check": cmd_check, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
