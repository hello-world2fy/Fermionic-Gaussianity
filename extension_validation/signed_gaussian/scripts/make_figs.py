"""Deterministic figures for Experiment C (fixed seed, no network, no fonts beyond default)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)

# ---- Fig 1: white noise curve ----
import csv
with open(os.path.join(OUT, "magic_curve.csv")) as f:
    rows = list(csv.DictReader(f))
ps = np.array([r["p"] for r in rows], float)
rlp = np.array([r["R_LP"] for r in rows], float)
rexp = np.array([r["R_expected"] for r in rows], float)
rwit = np.array([r["witness_lower"] for r in rows], float)

p_line = np.linspace(0, 1, 200)
r_line = np.maximum(1.0, 3.0 - 7.0 * p_line / 2.0)
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(p_line, r_line, "k-", lw=1.5, label="analytic $\\max(1,\\,3-7p/2)$")
ax.plot(ps, rlp, "ro", ms=6, label="LP dictionary $R_{LP}$")
ax.plot(ps, rwit, "b^", ms=5, mfc="none", label="witness lower bound $\\mathrm{Tr}(W\\rho_p)$")
ax.axvline(4/7, color="gray", ls="--", lw=1)
ax.text(4/7+0.01, 1.15, "$p_c=4/7$", color="gray")
ax.set_xlabel("noise strength $p$"); ax.set_ylabel("signed synthesis cost $R(\\rho_p)$")
ax.set_title("Four-mode magic state + even-parity white noise")
ax.legend(fontsize=8); ax.set_ylim(0, 3.3); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_white_noise_curve.png"), dpi=140)
plt.close(fig)

# ---- Fig 2: Hubbard R_norm heatmap ----
with open(os.path.join(OUT, "hubbard_curve.csv")) as f:
    hrows = list(csv.DictReader(f))
Us = sorted(set(float(r["U"]) for r in hrows))
betas = sorted(set(float(r["beta"]) for r in hrows))
Z = np.zeros((len(Us), len(betas)))
for r in hrows:
    Z[Us.index(float(r["U"])), betas.index(float(r["beta"]))] = r["R_norm"]
fig, ax = plt.subplots(figsize=(5.5, 4))
im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
               extent=[betas[0], betas[-1], Us[0], Us[-1]])
ax.set_xlabel("inverse temperature $\\beta$"); ax.set_ylabel("interaction $U$")
ax.set_title("Hubbard even-block signed cost $R_{norm}$ (dictionary upper bound)")
fig.colorbar(im, ax=ax, label="$R_{norm}$")
for i, U in enumerate(Us):
    for j, b in enumerate(betas):
        ax.text(b, U, f"{Z[i,j]:.2f}", ha="center", va="center",
                color="w", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_hubbard_heatmap.png"), dpi=140)
plt.close(fig)

# ---- Fig 3: witness finite-pool vs rigorous ----
with open(os.path.join(OUT, "hubbard_summary.json")) as f:
    hs = json.load(f)
w = hs["witness"]
fig, ax = plt.subplots(figsize=(5.5, 3.5))
labels = ["finite-pool\nratio", "rigorous $\\|W\\|$\nratio", "witness\nthreshold"]
vals = [w["ratio_pool"], w["ratio_rigorous_opnorm"], 1.0]
colors = ["#d62728", "#2ca02c", "gray"]
ax.bar(labels, vals, color=colors)
ax.axhline(1.0, color="gray", ls="--")
ax.set_ylabel("$\\mathrm{Tr}(W\\rho)/\\sup_g\\mathrm{Tr}(W\\sigma_g)$")
ax.set_title("Few-observable witness: pool suggests, continuous bound rejects")
for i, v in enumerate(vals):
    ax.text(i, v + 0.1, f"{v:.3f}", ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_witness_pool_vs_rigorous.png"), dpi=140)
plt.close(fig)

print("figures written to", FIG)
for fn in sorted(os.listdir(FIG)):
    print("  ", fn)
