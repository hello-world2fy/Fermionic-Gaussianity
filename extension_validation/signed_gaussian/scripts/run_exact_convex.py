"""Final bounded convex solve: FINITE-POOL exact J_k via SciPy L-BFGS / closed form.

Replaces the un-converged Monte-Carlo SGD (KKT 0.04-0.11, kept in convex_summary.json
as a failed-convergence attempt).  Here we fix a finite Gaussian pool and optimize the
exact, deterministic objective
    J(u) = (1/(2k)) (1/M) sum_j |<u, f_j>|^{2k} - <u, y>
where f_j = HS-flattened pool atoms sigma_j, y = flatten(target rho), and the reference
measure is the simple empirical one p_j = 1/M (NOT the spinor Weyl dimension d_{n,k};
for operators any positive d is a gauge rescaling and has no replica-dimension meaning).

Outputs: k=1 (closed form, normal equations) and k=2 (L-BFGS) over the full Hermitian
span; projected KKT residual in the pooled metric; full-target operator error; a
feature-restricted solve u = c0 I + cD(Da+Db) + cH hop; signed-coeff L1; and the
finite-pool LP lower/upper bounds (a finite-pool extent, NOT a continuum certificate).
The moment norm N_k(u) and sampling-sup normalization are DIAGNOSTIC only.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import minimize

from sggen.gaussian import (magic_ket, density_matrix, contact_kets,
                            product_family_kets, sample_even_gaussian_rays,
                            gaussian_ket_even, EVEN_OCC)
from sggen.synthesis import solve_signed_lp, flatten_hermitian, unflatten_hermitian
from sggen.hubbard import hubbard_hamiltonian, gibbs_state, even_block, annihilator, AUP, ADOWN, BUP, BDOWN

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
SEED = 20261026
rng = np.random.default_rng(SEED)
t0 = time.time()


from sggen.audit import audit_ket

def build_pool(M=600):
    kets = []
    kets += contact_kets(rng, 250)
    kets += product_family_kets(np.linspace(0.02, 8.0, 50))
    kets += list(sample_even_gaussian_rays(rng, 250))
    kets = kets[:M]
    for k in kets:                       # FAIL-CLOSED audit entry
        audit_ket(k, label="feature_pool")
    return np.array([density_matrix(k) for k in kets]), np.array(kets)


def frame_svd(atoms):
    """f_j = HS-flattened atoms; return orthonormal span basis via SVD."""
    F = np.array([flatten_hermitian(a) for a in atoms])   # (M, 64)
    # Gram-whiten: basis of the row span
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    keep = S > 1e-10 * S[0]
    basis = Vt[keep]                       # (r, 64), orthonormal rows
    return F, basis, S[keep]


def solve_case(y, F, basis, k, label):
    """Minimize J in the span coords z (u = basis.T z).  L-BFGS for k=2, closed k=1."""
    # project y into span coords
    yb = basis @ y                          # (r,)
    # frame matrix B = F @ basis.T  (M, r) : <u,f_j> = z @ (F@basis.T)[j]
    B = F @ basis.T
    M = F.shape[0]
    if k == 1:
        # J = (1/2) z^T G z - z^T yb, G = (1/M) B^T B ; normal equations
        G = (B.T @ B) / M
        z = np.linalg.solve(G + 1e-12 * np.eye(len(yb)), yb)
        iters = 0
    else:
        def obj(z):
            a = B @ z                       # <u,f_j>
            return (np.sum(np.abs(a) ** (2 * k)) / (2 * k) / M) - z @ yb
        def grad(z):
            a = B @ z
            c = (np.abs(a) ** (2 * k - 2)) * a / M
            return B.T @ c - yb
        res = minimize(obj, np.zeros(len(yb)), jac=grad, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
        z = res.x; iters = res.nit
    u = basis.T @ z
    # KKT: Q(u) := (1/M) sum c_j f_j must equal y (the target).  Reconstruct Q in
    # full 64-space and measure ||Q(u)-rho||; u itself is the dual witness, not rho.
    a = B @ z
    c = (np.abs(a) ** (2 * k - 2)) * a / M
    grad_span = B.T @ c - yb
    kkt_proj = float(np.linalg.norm(grad_span))
    Q_full = basis.T @ (B.T @ c)          # reconstructed operator (flattened)
    full_err = float(np.linalg.norm(Q_full - y))
    # finite-pool sup (diagnostic) and finite-pool witness value
    pool_sup = float(np.max(np.abs(a)))
    witness_val = float(u @ y)
    finite_lower = witness_val / pool_sup if pool_sup > 0 else float("nan")
    print(f"[{label} k={k}] iters={iters} KKT_proj={kkt_proj:.2e} "
          f"||Q(u)-rho||={full_err:.3e} pool_sup={pool_sup:.4f} finite_lower={finite_lower:.4f}")
    return {"k": k, "iters": iters, "kkt_proj": kkt_proj, "recon_error_Q_minus_rho": full_err,
            "pool_sup": pool_sup, "witness_value": witness_val,
            "finite_pool_lower": finite_lower, "u": u}


# ============ CASE 1: rho_M full Hermitian ============
print("=== CASE rho_M (full Hermitian, finite pool) ===")
atoms, pool_kets = build_pool(600)
F, basis, sv = frame_svd(atoms)
print(f"pool M={len(atoms)}  span rank={basis.shape[0]} (of 64)")
y_M = flatten_hermitian(density_matrix(magic_ket()))
res_M = {}
for k in (1, 2):
    res_M[k] = solve_case(y_M, F, basis, k, "rhoM")

# LP finite-pool upper bound on the SAME pool
lp_M = solve_signed_lp(unflatten_hermitian(y_M), atoms)
print(f"[rhoM] LP finite-pool upper = {lp_M['value']:.6f} (residual {lp_M['residual']:.1e})")

# ============ CASE 2: feature-restricted on Hubbard star ============
print("\n=== CASE Hubbard star (feature span 1+Da+Db+hop) ===")
c = [annihilator(j) for j in range(4)]
cd = [x.conj().T for x in c]; nn = [cd[j] @ c[j] for j in range(4)]
D_a = nn[AUP] @ nn[ADOWN]; D_b = nn[BUP] @ nn[BDOWN]
hop = cd[AUP]@c[BUP] + cd[ADOWN]@c[BDOWN] + cd[BUP]@c[AUP] + cd[BDOWN]@c[ADOWN]
def er(O):
    return O[np.ix_(list(EVEN_OCC), list(EVEN_OCC))]
I8 = np.eye(8, dtype=complex); Dsum = er(D_a) + er(D_b); hop_e = er(hop)
H = hubbard_hamiltonian(t_hop=1.0, U=8.0, mu=4.0)
rho_star = even_block(gibbs_state(H, 8.0)); rho_star /= np.trace(rho_star)
y_star = flatten_hermitian(rho_star)
feats = np.array([flatten_hermitian(I8), flatten_hermitian(Dsum), flatten_hermitian(hop_e)])
# PHYSICS FIX: flatten_hermitian stores raw diag/Re/Im, but Tr(O rho) carries a factor 2 on
# every off-diagonal HS pair.  For OBSERVATION covectors the off-diagonal entries must be x2
# so that feats @ flatten(rho) == actual Tr(O rho).  (Full-state synthesis coordinates stay
# injective/unweighted; only the feature covectors are corrected.)
def hs_covector(v):
    w = v.copy()
    w[8:] *= 2.0
    return w
feats = np.array([hs_covector(f) for f in feats])
hop_actual = float(np.trace(hop_e @ rho_star).real)
hop_covec = float(feats[2] @ flatten_hermitian(rho_star))
print(f"[sanity] Tr(hop rho_star)={hop_actual:.6f}  covector dot={hop_covec:.6f}  match={abs(hop_actual-hop_covec)<1e-10}")

# feature solve: u = feats.T c ; J = (1/2kM) sum |(feats c).f_j|^{2k} - (feats c).y
Bf = F @ feats.T     # (M, 3)
yb_f = feats @ y_star
feat_res = {}
for k in (1, 2):
    if k == 1:
        G = (Bf.T @ Bf) / len(atoms)
        cc = np.linalg.solve(G + 1e-12 * np.eye(3), yb_f)
        it = 0
    else:
        def obj(cv):
            a = Bf @ cv
            return np.sum(np.abs(a) ** (2*k))/(2*k)/len(atoms) - cv @ yb_f
        def grad(cv):
            a = Bf @ cv
            return Bf.T @ ((np.abs(a)**(2*k-2))*a/len(atoms)) - yb_f
        rr = minimize(obj, np.zeros(3), jac=grad, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
        cc = rr.x; it = rr.nit
    a = Bf @ cc
    gradf = Bf.T @ ((np.abs(a)**(2*k-2))*a/len(atoms)) - yb_f
    kkt = float(np.linalg.norm(gradf))
    # frame reconstruction coefficients c_j = |a_j|^{2k-2} a_j / M ; their L1 is the signed mass
    cj = (np.abs(a)**(2*k-2))*a/len(atoms)
    signed_mass = float(np.sum(np.abs(cj)))
    feat_err = float(np.linalg.norm(Bf.T @ cj - yb_f))
    u_star = feats.T @ cc
    dual_norm = float(np.max(np.abs(a)))
    print(f"[hubbard-feat k={k}] iters={it} KKT={kkt:.2e} coeff={np.round(cc,4)} "
          f"signed_massΣ|c_j|={signed_mass:.4f} feat_recon_err={feat_err:.3e} "
          f"dual_sup(diagnostic)={dual_norm:.4f}")
    feat_res[k] = {"iters": it, "kkt": kkt, "coeff": cc.tolist(),
                   "signed_mass": signed_mass, "feat_recon_error": feat_err,
                   "dual_sup_diagnostic": dual_norm, "cj": cj, "a": a, "u": u_star}

# ---- export feature NPZ (all keys for independent re-computation) ----
np.savez_compressed(os.path.join(OUT, "feature_attempt.npz"),
                    pool_kets=pool_kets, atoms=atoms,
                    feature_matrix_Bf=Bf, target_features_yb=yb_f,
                    obs_I=I8, obs_Dsum=Dsum, obs_hop=hop_e,
                    target_rho=rho_star,
                    cj_k1=feat_res[1]["cj"], cj_k2=feat_res[2]["cj"],
                    recon_features_k1=Bf.T @ feat_res[1]["cj"],
                    recon_features_k2=Bf.T @ feat_res[2]["cj"],
                    Wdual_k1=feat_res[1]["u"], Wdual_k2=feat_res[2]["u"],
                    signed_mass_k1=feat_res[1]["signed_mass"],
                    signed_mass_k2=feat_res[2]["signed_mass"],
                    coeff_k1=feat_res[1]["coeff"], coeff_k2=feat_res[2]["coeff"],
                    hop_actual=hop_actual, seed=SEED)
print("[export] feature_attempt.npz saved")

lp_star = solve_signed_lp(rho_star, atoms)
print(f"[hubbard-star] LP finite-pool upper = {lp_star['value']:.4f} (residual {lp_star['residual']:.1e})")

out = {
    "method": "fixed finite Gaussian pool (M=%d), exact J_k L-BFGS/closed, p_j=1/M" % len(atoms),
    "pool_span_rank": int(basis.shape[0]),
    "rho_M": {str(k): {kk: vv for kk, vv in res_M[k].items() if kk != "u"} for k in res_M},
    "rho_M_LP_finite_upper": lp_M["value"],
    "rho_M_LP_residual": lp_M["residual"],
    "hubbard_feature": {str(k): {kk: vv for kk, vv in feat_res[k].items()
                                 if kk not in ("cj", "a", "u")} for k in feat_res},
    "hubbard_LP_finite_upper": lp_star["value"],
    "diagnostic_note": ("finite_pool_lower is a lower bound on the FINITE-POOL extent, not a "
                        "continuum certificate. Moment norm and sampling-sup normalization are "
                        "diagnostic only. Old MC SGD attempt (convex_summary.json) is failed "
                        "convergence, not a theorem."),
    "walltime_s": time.time() - t0, "seed": SEED,
}
with open(os.path.join(OUT, "exact_convex_summary.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[done] walltime {time.time()-t0:.2f}s")
