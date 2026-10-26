"""Regenerate figures from results/."""
import csv
import collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)

rows = list(csv.DictReader(open(RES / "benchmark_long.csv")))
cases = [("L4_U1_weak", "L=4, U=1 (weak)"), ("L4_U8_strong", "L=4, U=8 (strong)"),
         ("L6_U1_weak", "L=6, U=1 (weak)"), ("L6_U8_strong", "L=6, U=8 (strong)")]
methods = [("fock_greedy", "Fock greedy", "C0", "-"),
           ("pt2", "PT2 (E-N)", "C1", "-"),
           ("noci_greedy", "non-orth greedy", "C2", "--"),
           ("replica", "replica convex-J", "C3", "-"),
           ("power_weighted_heuristic", "power heuristic (ablation)", "gray", ":")]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (tag, title) in zip(axes.flat, cases):
    for m, lab, c, ls in methods:
        d = collections.defaultdict(list)
        for r in rows:
            if r["tag"] == tag and r["method"] == m:
                d[int(r["n_det"])].append(float(r["dE"]))
        ns = sorted(d); med = [np.median(d[n]) for n in ns]
        ax.plot(ns, med, ls, color=c, label=lab, lw=1.6)
    ax.set_yscale("log"); ax.set_xlabel("determinants in Ritz space")
    ax.set_ylabel("|E - E_exact|"); ax.set_title(title); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle("Matched-budget energy error (median over 3 seeds)")
fig.tight_layout(); fig.savefig(ART / "convergence_energy.png", dpi=130); plt.close(fig)

# walltime breakdown
fig, ax = plt.subplots(figsize=(8, 5))
end = collections.defaultdict(float)
for r in rows:
    k = (r["tag"], r["method"])
    if float(r["t_cumulative"]) > end[k]:
        end[k] = float(r["t_cumulative"])
x = np.arange(len(cases)); w = 0.15
for i, (m, lab, c, ls) in enumerate(methods):
    vals = [end[(tag, m)] for tag, _ in cases]
    ax.bar(x + i * w, vals, w, label=lab, color=c)
ax.set_xticks(x + 2 * w); ax.set_xticklabels([t for t, _ in cases], fontsize=8)
ax.set_ylabel("total walltime (s)"); ax.set_title("Cumulative walltime incl. pool/residual/optimize/Ritz")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(ART / "walltime.png", dpi=130); plt.close(fig)

# dimer extent
d = list(csv.DictReader(open(RES / "dimer_extent.csv")))
Us = [float(r["U"]) for r in d]
xi = [float(r["xi_analytic"]) for r in d]
xin = [float(r["xi_numerical"]) for r in d]
fig, ax = plt.subplots(figsize=(7, 4.5))
u = np.linspace(0, 20, 200); ax.plot(u, [1 + uu / np.sqrt(uu**2 + 16) for uu in u],
                                    "b-", label=r"analytic $\xi_S=1+U/\sqrt{U^2+16t^2}$")
ax.plot(Us, xin, "ro", ms=5, label="Takagi numerical")
ax.set_xlabel("U/t"); ax.set_ylabel(r"$\xi_S$"); ax.legend(); ax.set_title("Hubbard dimer: Takagi extent vs analytic")
ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(ART / "dimer_extent.png", dpi=130); plt.close(fig)

# H2 gauge
h2 = list(csv.DictReader(open(RES / "h2_demo.csv")))
Rs = [float(r["R"]) for r in h2]
g = [float(r["gauge_fci"]) for r in h2]
pr = [float(r["proj_fci"]) for r in h2]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(Rs, g, "bo-", label=r"gauge $\tau=\sum|c_a|$ (FCI state)")
ax.plot(Rs, pr, "rs--", label="proj residual (dictionary span gap)")
ax.set_xlabel("R (Angstrom)"); ax.set_ylabel("value")
ax.set_title("H2/STO-3G: convex-J gauge on a local non-orth dictionary")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(ART / "h2_gauge.png", dpi=130); plt.close(fig)
print("figures written")
