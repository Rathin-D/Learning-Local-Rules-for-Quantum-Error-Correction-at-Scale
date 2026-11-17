"""Figures of the pilot results, drawn only from the files in data/ (no simulation).

    python figures/make_figures.py      -> figures/pilot_results.png and .pdf
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # one hue, light -> dark = small -> large L
BLUE, INK, MUTED, GRID = "#2a78d6", "#1f1f1e", "#6b6b67", "#e6e5e1"


def rows(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK, "legend.frameon": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) small lattices: failure falls with size up to p = 1.8%
    small = rows("small_lattices_strict_2L.csv")
    p = [float(r["p_percent"]) for r in small]
    for col, L in zip(RAMP, (5, 7, 9, 13, 19)):
        a.plot(p, [float(r[f"L{L}"]) for r in small], "-o", color=col, lw=2, ms=6,
               markeredgecolor="#fcfcfb", markeredgewidth=1, label=f"L = {L}")
    a.set_yscale("log")
    a.set_xlabel("noise per round, p (%)  [data and readings]")
    a.set_ylabel("strict failure rate")
    a.set_title("(a) Small lattices: failure falls with size up to p = 1.8%", fontsize=10, color=INK, loc="left")
    a.legend(loc="lower right")
    style(a)

    # (b) p = 1.6%: failure versus lattice size
    Ls = [5, 7, 9, 13, 19, 27, 39]
    r16 = next(r for r in small if r["p_percent"] == "1.6")
    b.plot(Ls, [float(r16[f"L{L}"]) for L in Ls], "o", color=BLUE, ms=7, mfc="#fcfcfb", mew=2,
           label="L = 5-39: cleanup up to 2L")
    large = rows("large_lattices_p1.6.csv")
    xl = [int(r["L"]) for r in large]
    k = [int(r["strict_20L"]) for r in large]
    n = [int(r["runs"]) for r in large]
    y = [ki / ni for ki, ni in zip(k, n)]
    lo = [y[i] - wilson(k[i], n[i])[0] for i in range(3)]
    hi = [wilson(k[i], n[i])[1] - y[i] for i in range(3)]
    b.errorbar(xl, y, yerr=[lo, hi], fmt="o", color=BLUE, ms=7, capsize=3, lw=1.5,
               label="L = 79-155: cleanup 20L (95% Wilson)")
    for x, yy, ki, ni in zip(xl, y, k, n):
        b.annotate(f"{ki:,}/{ni:,}", (x, yy), textcoords="offset points", xytext=(-8, -3), fontsize=8, color=MUTED,
                   ha="right")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xticks([5, 9, 19, 39, 79, 155])
    b.set_xticklabels(["5", "9", "19", "39", "79", "155"])
    b.set_xlabel("lattice size L")
    b.set_ylabel("strict failure rate at p = 1.6%")
    b.set_title("(b) p = 1.6%: suppression to L = 39, reversal beyond", fontsize=10, color=INK, loc="left")
    b.legend(loc="upper center", bbox_to_anchor=(0.42, 1.0), fontsize=9)
    style(b)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"pilot_results.{ext}", dpi=200)
    print("wrote", HERE / "pilot_results.png")


if __name__ == "__main__":
    main()
