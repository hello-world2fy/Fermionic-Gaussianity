"""Real convex program J_k(W) for the operator extent dual witness (k=1,2,4).

NOT an LP.  Solves J_k(W)=d_{n,k}/(2k) E|Tr(W sigma_g)|^{2k} - Tr(W rho) by
Monte-Carlo projected gradient, verifies the gradient by finite differences, and
reports the KKT reconstruction C_k(W_k)=rho.  The k-th-moment gauge witness
value Tr(W rho)/N_k(W) is reported alongside the numerically estimated true
sup-gauge Tr(W rho)/sup_g|Tr(W sigma_g)|; the latter is the honest lower bound.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sggen.gaussian import magic_ket, density_matrix, sample_even_gaussian_rays
from sggen.convex import (objective_and_grad, solve_convex, herm_to_real,
                          d_nk, N_MODES, spin_to_fock)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
SEED = 20261026
rng = np.random.default_rng(SEED)
t0 = time.time()
rhoM = density_matrix(magic_ket())

# ---- gradient finite-difference verification on fixed rays ----
rays_fix = sample_even_gaussian_rays(np.random.default_rng(5), 6000)
x0 = herm_to_real(rhoM) * 0.3
grad_check = {}
for k in [1, 2, 4]:
    J, gx = objective_and_grad(x0, rhoM, rays_fix, k)
    err = 0.0
    for idx in [0, 10, 33, 50]:
        eps = 1e-6
        Jp, _ = objective_and_grad(x0 + eps * np.eye(64)[idx], rhoM, rays_fix, k)
        Jm, _ = objective_and_grad(x0 - eps * np.eye(64)[idx], rhoM, rays_fix, k)
        err = max(err, abs((Jp - Jm) / (2 * eps) - gx[idx]))
    grad_check[str(k)] = float(err)
    print(f"[grad-check k={k}] finite-diff max err = {err:.2e}")

# ---- solve each k ----
rays_sup = sample_even_gaussian_rays(np.random.default_rng(99), 200000)
results = {}
for k in [1, 2, 4]:
    res = solve_convex(rhoM, k, seed=SEED, iterations=1200, batch=4096)
    Wk = res["Wk"]
    t = np.abs(np.einsum("bi,ij,bj->b", rays_sup.conj(), Wk, rays_sup))
    h = float(t.max())
    num = float(np.trace(Wk @ rhoM).real)
    results[str(k)] = {
        "d_nk": d_nk(N_MODES, k),
        "KKT_residual": float(res["kkt_residual"]),
        "Tr_W_rho": num,
        "moment_gauge_Nk": float(res["Nk"]),
        "moment_witness": float(res["witness_lower"]),
        "numerical_sup_h": h,
        "sup_gauge_lower_bound": num / h,
    }
    print(f"[k={k}] KKT={res['kkt_residual']:.3e} moment_witness={res['witness_lower']:.4f} "
          f"sup_gauge_lower={num/h:.4f}")

out = {
    "program": "J_k(W)=d/(2k)E|Tr(W sigma_g)|^{2k}-Tr(W rho), Monte-Carlo projected gradient",
    "gradient_finite_diff_max_err": grad_check,
    "note": ("moment_gauge uses N_k=(d E|t|^{2k})^{1/(2k)}, which is <= d^{1/(2k)} h and so "
             "is NOT a certified sup-normalized witness; sup_gauge_lower_bound uses the "
             "numerical continuous sup over 2e5 rays. The analytic witness W=4rhoM-P+ gives "
             "the certified lower bound 3 (see magic_summary)."),
    "results": results,
    "walltime_s": time.time() - t0, "seed": SEED,
}
with open(os.path.join(OUT, "convex_summary.json"), "w") as f:
    json.dump(out, f, indent=2)
print("[done] walltime", time.time() - t0)
