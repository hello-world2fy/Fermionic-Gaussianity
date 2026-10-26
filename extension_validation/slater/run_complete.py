"""Fock-complete residual-coverage feasibility run (small budget 8/12)."""
import csv
import time
import warnings
from pathlib import Path

import numpy as np

from hubbard import HubbardModel
import noci
from complete_ci import run_complete

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def seeds(model):
    s = {0: model.neel_determinant()}
    m = 0
    for i in range(model.L):
        m |= (1 << (i if i % 2 else model.L + i))
    s[1] = m
    rng = np.random.default_rng(7)
    up = rng.choice(model.L, model.n_up, replace=False)
    dn = rng.choice(model.L, model.n_down, replace=False)
    m2 = 0
    for i in up: m2 |= 1 << int(i)
    for i in dn: m2 |= 1 << (model.L + int(i))
    s[2] = m2
    return s


def main():
    warnings.filterwarnings("ignore")
    cases = [("L4_U1", 4, 1.0), ("L4_U8", 4, 8.0),
             ("L6_U1", 6, 1.0), ("L6_U8", 6, 8.0)]
    rows = []
    t0 = time.perf_counter()
    methods = ["cipsi", "greedy_noci", "replica_coef", "replica_energy"]
    for tag, L, U in cases:
        model = HubbardModel(L=L, t=1.0, U=U)
        E_ex, _ = model.exact_ground()
        sd = seeds(model)
        for method in methods:
            for seed in [0, 1, 2]:
                recs = run_complete(model, sd[seed], method, budget=12,
                                    k=2.0, E_exact=E_ex)
                for r in recs:
                    rows.append(dict(
                        tag=tag, L=L, U=U, method=method, seed=seed,
                        n_det=r.n_det, E=r.E, dE=r.dE, residual=r.residual,
                        kkt_err=r.kkt_err, proj_residual=r.proj_residual,
                        gauge=r.gauge, picked_nonorth=r.picked_nonorth,
                        t_pool=r.t_pool, t_residual=r.t_residual,
                        t_optimize=r.t_optimize, t_ritz=r.t_ritz,
                        pool_size=r.pool_size))
                last = recs[-1]
                kk = "" if last.kkt_err != last.kkt_err else f"{last.kkt_err:.1e}"
                print(f"[{tag}] {method:13s} seed{seed}: dE@12={last.dE:.3e} "
                      f"kkt={kk} n_rows={len(recs)}")
    with open(RES / "complete_ci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[done] {time.perf_counter()-t0:.1f}s, {len(rows)} rows")
    # refresh summary.json from the freshly-written CSV (no experiment re-run)
    import make_summary
    make_summary.main()


if __name__ == "__main__":
    main()
