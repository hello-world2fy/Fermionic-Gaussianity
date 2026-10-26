"""Corrected end-to-end driver:
  * Hubbard L=4/6 half-filled, weak/strong U, 3 seeds, 4 selection methods
    (fock_greedy, pt2, noci_greedy, replica=real convex-J), matched budget.
  * H2/STO-3G feasibility synthesis demo at R=0.74/1.5/2.0 A.

Writes CSV/NPZ/JSON under results/.
"""
from __future__ import annotations

import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np

from hubbard import HubbardModel
from dimer_extent import dimer_extent_numerical, xi_S_analytic
from selected_ci import run_selected_ci
import noci

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)


def seeds_for(model):
    s = {}
    s[0] = model.neel_determinant()
    m_anti = 0
    for i in range(model.L):
        if i % 2 == 1:
            m_anti |= 1 << i
        else:
            m_anti |= 1 << (model.L + i)
    s[1] = m_anti
    rng = np.random.default_rng(2026)
    ups = rng.choice(model.L, size=model.n_up, replace=False)
    dns = rng.choice(model.L, size=model.n_down, replace=False)
    m2 = 0
    for i in ups:
        m2 |= 1 << int(i)
    for i in dns:
        m2 |= 1 << (model.L + int(i))
    s[2] = m2
    return s


def run_hubbard():
    cases = [
        ("L4_U1_weak", 4, 1.0), ("L4_U8_strong", 4, 8.0),
        ("L6_U1_weak", 6, 1.0), ("L6_U8_strong", 6, 8.0),
    ]
    methods = ["fock_greedy", "pt2", "noci_greedy", "replica",
               "power_weighted_heuristic"]
    max_det = {4: 36, 6: 200}
    rows = []
    exact = {}
    for tag, L, U in cases:
        model = HubbardModel(L=L, t=1.0, U=U)
        E_ex, psi_ex = model.exact_ground()
        exact[tag] = dict(E_exact=float(E_ex), dim=int(model.dim))
        seeds = seeds_for(model)
        for method in methods:
            krep_list = [2.0] if method == "replica" else [None]
            for kr in krep_list:
                for s in [0, 1, 2]:
                    res = run_selected_ci(
                        model, seeds[s], method, k_new=2,
                        max_det=max_det[L], k_rep=(kr or 2.0),
                        E_exact=E_ex, psi_exact=psi_ex,
                        n_atoms=40, dict_eps=0.6)
                    tc = 0.0
                    for rec in res["records"]:
                        tc += rec.t_pool + rec.t_residual + rec.t_optimize + rec.t_ritz
                        rows.append(dict(
                            tag=tag, L=L, U=U, method=method,
                            k_rep=(kr if kr is not None else ""), seed=s,
                            n_det=rec.n_det, E=rec.E, dE=rec.dE,
                            residual=rec.residual_norm, overlap=rec.overlap,
                            double_occ=rec.double_occ, pool_size=rec.pool_size,
                            t_pool=rec.t_pool, t_residual=rec.t_residual,
                            t_optimize=rec.t_optimize, t_ritz=rec.t_ritz,
                            t_cumulative=round(tc, 6),
                            kkt_err=("" if np.isnan(rec.kkt_err) else rec.kkt_err),
                            gauge=("" if np.isnan(rec.gauge) else rec.gauge),
                            proj_residual=("" if np.isnan(rec.proj_residual) else rec.proj_residual),
                            opt_iters=rec.opt_iters))
                    print(f"  [{tag}] {method} kr={kr} seed={s}: "
                          f"final dE={res['records'][-1].dE:.2e} n={res['records'][-1].n_det}")
    fields = list(rows[0].keys())
    with open(RES / "benchmark_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows, exact


def run_h2():
    """Feasibility demo: synthesize the H2 ground state (and the RHF residual)
    on a non-orthogonal Slater dictionary; report KKT residual and gauge tau."""
    from molecule import h2_active
    out = []
    for R in [0.74, 1.5, 2.0]:
        warnings.filterwarnings("ignore")
        ham, fci_e, rhf_e, mf = h2_active(R)
        E, psi = ham.exact_ground()
        r0 = ham.neel_determinant()
        guess = np.zeros(ham.dim); guess[ham.index[r0]] = 1.0
        # 1) gauge of the FCI state itself on a non-orth dictionary
        A = noci.sample_dictionary(ham, r0, n_atoms=20, eps=0.8, seed=0)
        syn = noci.convex_j_synthesize(A, psi, k=2.0)
        # 2) RHF residual synthesis
        r = ham.H @ guess - E * guess
        r /= np.linalg.norm(r)
        syn_r = noci.convex_j_synthesize(A, r, k=2.0)
        out.append(dict(R=R, E_FCI=E, E_RHF=rhf_e, nuc=float(mf.energy_nuc()),
                        gauge_fci=syn["gauge"], kkt_fci=syn["kkt_err"],
                        proj_fci=syn["proj_residual"],
                        gauge_res=syn_r["gauge"], kkt_res=syn_r["kkt_err"],
                        proj_res=syn_r["proj_residual"],
                        span_dim=syn["span_dim"]))
        print(f"  [H2 R={R}] E_FCI={E:.4f} gauge(FCI)={syn['gauge']:.3f} "
              f"kkt={syn['kkt_err']:.1e} | residual kkt={syn_r['kkt_err']:.1e}")
    with open(RES / "h2_demo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    return out


def main():
    t0 = time.perf_counter()
    # dimer extent
    drows = []
    for U in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0]:
        r = dimer_extent_numerical(U, 1.0); xa = xi_S_analytic(U, 1.0)
        drows.append(dict(U=U, xi_numerical=r["xi_S"], xi_analytic=xa,
                          relerr=abs(r["xi_S"]-xa)/xa, lam1=r["lam1"], lam2=r["lam2"]))
    with open(RES / "dimer_extent.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0].keys()))
        w.writeheader(); w.writerows(drows)
    print(f"[dimer] max relerr={max(r['relerr'] for r in drows):.2e}")

    rows, exact = run_hubbard()
    h2 = run_h2()
    np.savez(RES / "exact_states.npz",
             **{f"{t}_E": v["E_exact"] for t, v in exact.items()})
    params = dict(walltime=time.perf_counter()-t0, n_rows=len(rows),
                  exact=exact, h2=h2,
                  note="t=1; open chain; Sz=0; k_new=2; methods fock_greedy/pt2/"
                       "noci_greedy/replica(real convex-J); power_weighted_heuristic is ablation")
    json.dump(params, open(RES / "run_params.json", "w"), indent=2)
    print(f"[done] {time.perf_counter()-t0:.1f}s, {len(rows)} rows")


if __name__ == "__main__":
    main()
