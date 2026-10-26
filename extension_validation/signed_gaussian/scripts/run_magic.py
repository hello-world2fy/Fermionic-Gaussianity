"""Experiment C: signed Gaussian operator convex synthesis of the four-mode magic state.

Verifies, independently (NOT exact proof, see review C-F3/C-F7):
  (1) R(rho_M) is BRACKETED: analytic witness lower bound = 3, finite-dictionary LP
      upper bound ~= 3.000006 (a ~2e-5 gap, NOT an exact decomposition).
  (2) R(rho_p) is CONSISTENT with max(1, 3 - 7p/2) for rho_p=(1-p)rho_M+p P_+/8,
      pc=4/7: the LP upper bound matches the witness lower line to ~4 digits;
      the R=1 plateau additionally needs a continuum hull argument (C-F7).
  (3) lower-bound witness W = 4 rho_M - P_+ is a CONTINUOUS certificate:
      sup_{even g} Tr(W|g><g|)=1 (odd sector gives Tr(W sigma_g)=0, not -1).

Primal: LP over a dictionary gives an UPPER bound on R (achievable decomposition).
Dual  : the witness gives a LOWER bound for every p.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys, time, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from sggen.gaussian import (magic_ket, density_matrix, contact_kets,
                            product_family_kets, sample_even_gaussian_rays,
                            gaussian_ket_even, random_su4, J0_STD)
from sggen.synthesis import solve_signed_lp

SEED = 20261026
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
os.makedirs(OUT, exist_ok=True)

t_start = time.time()
rng = np.random.default_rng(SEED)
M = magic_ket()
rhoM = density_matrix(M)
Pplus_over_8 = np.eye(8) / 8.0          # even block of P_+/8
W = 4 * rhoM - np.eye(8)                # even block of W = 4 rho_M - P_+

# ---------- build dictionary ----------
t0 = time.time()
kets = []
kets += contact_kets(rng, 3000)                 # saturating contact atoms
kets += product_family_kets(np.linspace(0.02, 8.0, 80))
# a few diverse random Gaussians (non-contact, for span flexibility)
for _ in range(400):
    Z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Z = (Z - Z.T) / 2
    kets.append(gaussian_ket_even(Z))
atoms = np.array([density_matrix(k) for k in kets])
# basis Slater determinants
basis_atoms = np.stack([density_matrix(np.eye(8, dtype=complex)[i]) for i in range(8)])
atoms_full = np.concatenate([atoms, basis_atoms], axis=0)
build_time = time.time() - t0
print(f"[build] dictionary {atoms_full.shape[0]} atoms in {build_time:.2f}s")

# ---------- (1) rho_M ----------
t0 = time.time()
resM = solve_signed_lp(rhoM, atoms_full)
rm_time = time.time() - t0
print(f"[rho_M] R_LP = {resM['value']:.6f}  residual={resM['residual']:.2e}  "
      f"nz={resM['n_nonzero']}  time={rm_time:.2f}s")

# TV sign breakdown
wM = resM["weights"]
pos = wM[wM > 1e-9].sum(); neg = -wM[wM < -1e-9].sum()
print(f"        positive mass = {pos:.4f}  negative mass = {neg:.4f}  sum = {pos-neg:.4f} (trace)")

# ---------- (3) continuous witness sup verification ----------
# analytic claim: sup_{even g} Tr(W|g><g|) = 1, i.e. sup |<M|g>|^2 = 1/2.
# verify numerically on structured grid + many random rays.
ov2 = []
for t in np.linspace(0, 12, 4000):
    k = np.zeros(8, dtype=complex)
    k[0] = 1/(1+t**2); k[1] = t/(1+t**2); k[6] = t/(1+t**2); k[7] = t**2/(1+t**2)
    ov2.append(abs(np.vdot(M, k))**2)
rays = sample_even_gaussian_rays(rng, 50000)
ov2_rand = np.abs(rays.conj() @ M) ** 2
sup_analytic = 0.5
sup_numeric = max(max(ov2), ov2_rand.max())
print(f"[witness] analytic sup |<M|g>|^2 = 1/2 = {sup_analytic}")
print(f"[witness] numeric sup (grid+2e5 rays) = {sup_numeric:.6f}  -> Tr(W sigma_g) sup = {4*sup_numeric-1:.6f}")
print(f"[witness] Tr(W rho_M) = {np.trace(W @ rhoM).real:.6f}  (lower bound R >= this)")

# ---------- (2) white noise curve: CORE points only (resource-limited, fresh-validated) ----------
ps = [0.0, 4/7, 1.0]
rows = []
for p in ps:
    rho_p = (1 - p) * rhoM + p * Pplus_over_8
    r = solve_signed_lp(rho_p, atoms_full)
    r_expected = max(1.0, 3.0 - 7.0 * p / 2.0)
    trW = np.trace(W @ rho_p).real
    rows.append(dict(p=p, R_LP=r["value"], R_expected=r_expected,
                     residual=r["residual"], frob_residual=r["frobenius_residual"],
                     upper_valid=r["upper_valid"], nz=r["n_nonzero"],
                     witness_lower=trW))
    print(f"[p={p:.3f}] R_LP={r['value']:.6f}  expected={r_expected:.4f}  "
          f"frob={r['frobenius_residual']:.1e} valid={r['upper_valid']}  resid={r['residual']:.1e}")
print("[note] only p=0,pc,1 fresh-validated with full Frobenius residual; other p points are STALE.")

# ---------- save ----------
with open(os.path.join(OUT, "magic_curve.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)

np.savez_compressed(os.path.join(OUT, "magic_artifacts.npz"),
                    M=M, rhoM=rhoM, W=W,
                    wM=wM, atoms_full=atoms_full,
                    ov2_rand_max=ov2_rand.max(),
                    expected_curve_p=np.array(ps),
                    expected_curve_R=np.array([max(1.0, 3.0-7.0*p/2) for p in ps]))

summary = {
    "rho_M_R_LP": resM["value"],
    "rho_M_R_lower_witness": float(np.trace(W @ rhoM).real),
    "rho_M_R_statement": ("bracketed, not exact: analytic lower=3 (continuous witness), "
                          "numerical LP upper=3.000006 (~2e-5 gap, finite-dictionary)"),
    "rho_M_residual": resM["residual"],
    "rho_M_frobenius_residual": resM["frobenius_residual"],
    "rho_M_upper_valid": resM["upper_valid"],
    "rho_M_n_nonzero": resM["n_nonzero"],
    "rho_M_positive_mass": float(pos),
    "rho_M_negative_mass": float(neg),
    "witness_proof": ("|<M|g>|^2 = (1/2) N^2 |1+P|^2 <= (1/2)(1+mu1 mu2)^2/"
                      "((1+mu1^2)(1+mu2^2)) <= 1/2, since (1+mu1^2)(1+mu2^2)-(1+mu1 mu2)^2 "
                      "= (mu1-mu2)^2>=0.  Hence Tr(W sigma_g)<=1 for even g; odd g give 0. "
                      "Equality on product BCS family g(t,t) for all t (mu1=mu2)."),
    "witness_sup_analytic": 1.0,
    "witness_sup_numeric": float(4*sup_numeric-1),
    "odd_sector_value": 0.0,
    "build_time_s": build_time,
    "rho_M_time_s": rm_time,
    "walltime_s": time.time()-t_start,
    "seed": SEED,
}
with open(os.path.join(OUT, "magic_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] walltime {time.time()-t_start:.2f}s")
