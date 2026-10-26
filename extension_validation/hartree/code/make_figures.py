"""Generate result figures into results/figures/ (matplotlib, Agg)."""
import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)


def fig_sandwich():
    rows = list(csv.DictReader(open(os.path.join(RES, "exp1_convergence.csv"))))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for n in sorted(set(r["n"] for r in rows)):
        sub = [r for r in rows if r["n"] == n and r["n"] in
               ["(2, 2)", "(1, 1, 1)", "(1, 1, 1, 1)", "(3, 3)"]]
        if not sub:
            continue
        k = [int(r["k"]) for r in sub]
        xi = float(sub[0]["xi"])
        bk = [float(r["Bk2"]) for r in sub]
        tk = [float(r["tk2"]) for r in sub]
        ax.plot(k, bk, "--o", ms=3, label=f"B_k^2 upper {n}")
        ax.plot(k, tk, ":s", ms=3, label=f"t_k^2 lower {n}")
        ax.axhline(xi, lw=0.8, color="gray")
    ax.set_xlabel("replica order k")
    ax.set_ylabel("coefficient cost / dual value")
    ax.set_title("Certified sandwich  t_k^2 <= xi <= B_k^2  (Fock states)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_sandwich.png"), dpi=140)
    plt.close(fig)


def fig_socp():
    d = json.load(open(os.path.join(RES, "exp2_convex.json")))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rows = d["rows"]
    seen = set()
    for r in rows:
        key = (r["m"], r["n"])
        if key in seen:
            continue
        seen.add(key)
        ax.scatter([r["tau_lower_explicit"]], [0], marker="_", s=200, color="k")
        ax.plot([r["tau_lower_explicit"], r["tau_socp_net2"]], [0, 0], lw=2,
                label=f"m={r['m']} {r['n']}")
    ax.set_xlabel("tau = sqrt(xi)")
    ax.set_title("Discrete-dictionary SOCP upper bound vs explicit lower bound\n"
                 "(line: certified interval; point: explicit witness lower bound)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_socp.png"), dpi=140)
    plt.close(fig)


def fig_bh():
    d = json.load(open(os.path.join(RES, "exp3_bosehubbard.json")))
    e4 = json.load(open(os.path.join(RES, "exp4_rdm.json")))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for rec in d["results"]:
        tag = f"M={rec['m']}"
        ws = rec["witness"]
        ks = [w["k"] for w in ws]
        axes[0].plot(ks, [w["xi_lower_certified"] for w in ws], ":o", label=tag+" lower(cert)")
        axes[0].plot(ks, [w["xi_upper_certified"] for w in ws], "--s", label=tag+" upper(cert)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("xi bounds")
    axes[0].set_title("BH ground state xi: certified interval\n(test-vector lower + Holder/residual upper)")
    axes[0].legend(fontsize=6)
    # condensation fraction (Penrose-Onsager lambda_max/N) vs F1 (Hartree fidelity)
    rows = e4["rows"]
    for m in [2, 3]:
        rr = [r for r in rows if r["m"] == m and r["J"] == 1.0]
        xs = [r["UoverJ"] for r in rr]
        axes[1].plot(xs, [r["condensate_fraction"] for r in rr], "o-", label=f"M={m} lambda_max/N")
        axes[1].plot(xs, [r["F1_hartree_estimate"] for r in rr], "x--", label=f"M={m} F1 (Hartree)")
    axes[1].set_xlabel("U/J"); axes[1].set_ylabel("fraction")
    axes[1].set_title("condensate fraction (1-RDM) vs\nF1 max Hartree fidelity (different metrics)")
    axes[1].legend(fontsize=6)
    axes[2].plot(xs, [r["entropy_site1_bits"] for r in rr], "o-", color="C1")
    axes[2].set_xlabel("U/J"); axes[2].set_ylabel("one-site entropy (bits)")
    axes[2].set_title("Many-body entanglement (Hartree = 0)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_bosehubbard.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_sandwich()
    fig_socp()
    try:
        fig_bh()
    except Exception as e:
        print("fig_bh skipped:", e)
    print("figures in", FIG)
