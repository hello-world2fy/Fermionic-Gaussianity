"""Experiment C (Hubbard): signed Gaussian decomposition of the two-site Hubbard Gibbs state.

Scope and honesty rules (per independent review c_findings.md):
  * The Gibbs state is GRAND-CANONICAL: H includes -mu(n_a+n_b), mu=U/2, and
    rho = e^{-beta H}/Z sums over all 16 Fock states (N=0..4).  Only the
    even-parity block (N=0,2,4) is decomposed.  R_norm = TV/Tr(rho_even).
  * beta=0.5 is NOT the maximally mixed state (beta != 0); R_norm=1 there means
    the LP recovered convex-Gaussian atoms exactly, not that rho=I.
  * The U=0 even block is a convex mixture of pure Slater-determinant Gaussians,
    so true R_norm=1 for all beta.  We ADD the U=0 quadratic eigenstates to the
    dictionary as Slater atoms; without them a random-ray dictionary leaves a
    ~2x low-T discretization bias (review C-F4).  Any reported interaction effect
    is excess = R(U,beta)-R(U=0,beta), not the raw R(U,beta).
  * Finite-observable witness sup is a FINITE-POOL estimate that SHRINKS as the
    pool refines; it is not a continuous certificate.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys, time, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from sggen.gaussian import (density_matrix, contact_kets, product_family_kets,
                            sample_even_gaussian_rays, gaussian_ket_even, EVEN_OCC)
from sggen.synthesis import solve_signed_lp
from sggen.hubbard import (hubbard_hamiltonian, gibbs_state, even_block,
                           annihilator, AUP, ADOWN, BUP, BDOWN)

SEED = 20261027
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
t_start = time.time()
rng = np.random.default_rng(SEED)

# ---------- even-sector dictionary: LEGAL audited Gaussian atoms only ----------
# Every atom is audited fail-closed (norm=1, even quadric=0, rho PSD/trace=1) BEFORE LP.
# NO interacting-H or full-Fock-eigh eigenstates are used (degenerate-space vectors fail).
from sggen.gaussian import free_slater_atoms
from sggen.audit import audit_ket
import itertools as _it

FREE = free_slater_atoms()
for k in FREE: audit_ket(k, label="free_wedge")

def exact_u0_weights(beta, hop=1.0, mu=0.0):
    """Analytic positive weights on the 8 free wedge atoms for U=0 Gibbs even block.

    Single-particle energies: bonding orbitals e=-hop-mu (x2 spins), antibonding
    e=+hop-mu (x2).  Fermi occupation f_j=1/(1+e^{beta e_j}); even-sector renormalized.
    """
    e = np.array([-hop-mu, hop-mu, -hop-mu, hop-mu])
    f = 1.0/(1.0+np.exp(beta*e))
    w = []
    for nocc in (0, 2, 4):
        for J in _it.combinations(range(4), nocc):
            p = 1.0
            for j in range(4):
                p *= f[j] if j in J else (1.0-f[j])
            w.append(p)
    w = np.array(w); w /= w.sum()
    return w

uni = list(sample_even_gaussian_rays(rng, 2000))
contact = contact_kets(rng, 800)
prod = product_family_kets(np.linspace(0.02, 8.0, 40))
free = free_slater_atoms()
basis_e = [np.eye(8, dtype=complex)[i] for i in range(8)]

pool_kets_dict, pool_sources = [], []
for k in uni: pool_kets_dict.append(k); pool_sources.append("haar")
for k in contact: pool_kets_dict.append(k); pool_sources.append("contact")
for k in prod: pool_kets_dict.append(k); pool_sources.append("pairing_product")
for k in free: pool_kets_dict.append(k); pool_sources.append("free_singleparticle_wedge")
for k in basis_e: pool_kets_dict.append(k); pool_sources.append("slater_basis")

# FAIL-CLOSED audit: raise on first illegal atom
for k, s in zip(pool_kets_dict, pool_sources):
    audit_ket(k, label=s)
print(f"[audit] {len(pool_kets_dict)} atoms all legal (quadric=0,norm=1,PSD)")

atoms = np.array([density_matrix(k) for k in pool_kets_dict])
source_labels = np.array(pool_sources)
pool_kets_dict = np.array(pool_kets_dict)
print(f"[build] audited dictionary {atoms.shape[0]} atoms")

# dense pool for witness sup (also audited)
pool_kets = sample_even_gaussian_rays(rng, 100000)
for k in pool_kets: audit_ket(k, label="witness_pool")
pool_dm = np.einsum("bi,bj->bij", pool_kets, pool_kets.conj())

# ---------- observables in the even (8-dim) block ----------
# full-16 projectors, then restrict to even block
c = [annihilator(j) for j in range(4)]
cd = [x.conj().T for x in c]
n = [cd[j] @ c[j] for j in range(4)]
D_a = n[AUP] @ n[ADOWN]            # double occupancy on site a
D_b = n[BUP] @ n[BDOWN]
hop = cd[AUP] @ c[BUP] + cd[ADOWN] @ c[BDOWN] + cd[BUP] @ c[AUP] + cd[BDOWN] @ c[ADOWN]

def even_restrict(O16):
    idx = list(EVEN_OCC)
    return O16[np.ix_(idx, idx)]

D_a_e = even_restrict(D_a); D_b_e = even_restrict(D_b); hop_e = even_restrict(hop)
OBS = {"D_a": D_a_e, "D_b": D_b_e, "hop": hop_e}

# ---------- grid: core points (fresh, audited pool) ----------
betas = [0.5, 1.0, 2.0, 8.0]
Us = [0.0, 2.0, 4.0, 8.0]
rows = []
records = []   # per-point: weights, target_rho, reconstructed, frob
u0_baseline = {}
for U in Us:
    H = hubbard_hamiltonian(t_hop=1.0, U=U, mu=U / 2.0)
    for beta in betas:
        rho16 = gibbs_state(H, beta)
        rho_e = even_block(rho16)
        tr_e = np.trace(rho_e).real
        rho_n = rho_e / tr_e
        if U == 0.0:
            # analytic exact positive decomposition on the 8 free wedge atoms
            w_exact = exact_u0_weights(beta, hop=1.0, mu=0.0)
            # map onto the shared pool indices where free atoms live
            free_idx = np.where(source_labels == "free_singleparticle_wedge")[0]
            assert len(free_idx) == 8, f"expected 8 free atoms, got {len(free_idx)}"
            weights_full = np.zeros(len(atoms))
            weights_full[free_idx] = w_exact
            recon = np.tensordot(weights_full, atoms, axes=(0, 0))
            frob = float(np.linalg.norm(recon - rho_n))
            R = 1.0; resid = frob; nz = int((weights_full > 1e-12).sum())
            u0_baseline[beta] = 1.0
            method = "analytic_free_wedge"
        else:
            r = solve_signed_lp(rho_n, atoms)
            R = r["value"]; resid = r["residual"]; frob = r["frobenius_residual"]; nz = r["n_nonzero"]
            recon = np.tensordot(r["weights"], atoms, axes=(0,0))
            weights_full = r["weights"]
            method = "lp"
        obs = {k: np.trace(O @ rho_n).real for k, O in OBS.items()}
        rows.append(dict(U=U, beta=beta, trace_even=tr_e,
                         R_norm=R, residual=resid, frob_residual=frob,
                         upper_valid=bool(frob<1e-8), nz=nz, method=method, **obs))
        records.append(dict(U=U, beta=beta, weights=weights_full,
                            target_rho=rho_n, reconstructed=recon, frob=frob))
        print(f"[U={U:4.1f} b={beta:4.1f}] R_norm={R:.4f} frob={frob:.1e} "
              f"valid={frob<1e-8} <D_a>={obs['D_a']:.3f} ({method})")
print("[sanity] U=0 baseline R_norm:", {b: round(u0_baseline[b], 4) for b in betas})
print("[note] excess over U=0 baseline is the interaction-induced, dictionary-dependent signal.")

# ---------- finite-observable witness at the most interacting point ----------
# W = w0 I + w_D (D_a+D_b) + w_h hop ; optimize ratio Tr(W rho)/sup_pool Tr(W sigma_g)
U_star, beta_star = 8.0, 8.0
H = hubbard_hamiltonian(t_hop=1.0, U=U_star, mu=U_star / 2)
rho_star = even_block(gibbs_state(H, beta_star))
rho_star = rho_star / np.trace(rho_star)

Dsum = D_a_e + D_b_e
best = None
# coarse grid over coefficients
for w0 in np.linspace(-2, 2, 17):
    for wD in np.linspace(-2, 2, 17):
        for wh in np.linspace(-1, 1, 9):
            Ww = w0 * np.eye(8) + wD * Dsum + wh * hop_e
            Ww = (Ww + Ww.conj().T) / 2.0          # enforce real Hermitian
            num = abs(np.trace(Ww @ rho_star).real)
            # correct trace: Tr(W sigma_g) = sum_ij sigma[i,j] W[j,i]
            tra = np.einsum("bij,ji->b", pool_dm, Ww).real
            sup_pool = np.max(np.abs(tra))
            if sup_pool <= 0:
                continue
            ratio = num / sup_pool
            if best is None or ratio > best[0]:
                best = (ratio, w0, wD, wh, num, sup_pool)
ratio, w0, wD, wh, num, sup_pool = best
Ww = w0 * np.eye(8) + wD * Dsum + wh * hop_e
Ww = (Ww + Ww.conj().T) / 2.0
# continuous-ish sup on product family (abs, correct trace)
extra = []
for t in np.linspace(0.02, 8.0, 200):
    k = np.zeros(8, dtype=complex)
    k[0] = 1/(1+t**2); k[1] = t/(1+t**2); k[6] = t/(1+t**2); k[7] = t**2/(1+t**2)
    extra.append(abs(np.real(k.conj() @ Ww @ k)))
sup_grid = max(sup_pool, max(extra))
opnorm = np.linalg.norm(Ww, 2)
print(f"\n[witness U=8,b=8] best ratio(pool)= {ratio:.4f} at w0={w0:.2f} wD={wD:.2f} wh={wh:.2f}")
print(f"   |Tr(W rho)|={num:.4f}  sup_pool={sup_pool:.4f}  sup_grid={sup_grid:.4f}")
print(f"   -> finite-pool ratio = {num/sup_pool:.3f} (OPTIMISTIC, pool is a subset)")
print(f"   -> rigorous ||W|| bound = {opnorm:.4f} -> certified ratio <= {num/opnorm:.3f}")
print(f"   -> NOTE: ||W||/rho ratio <= 1 for ANY density; this is a coarse, UNINFORMATIVE")
print(f"      bound, i.e. a FAILED certificate, NOT evidence that no witness exists.")

with open(os.path.join(OUT, "hubbard_curve.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)

# record star-point LP weights / reconstruction for audit
# per-point curve records (shared pool keys)
np.savez_compressed(os.path.join(OUT, "hubbard_curve_records.npz"),
                    params=np.array([[d["U"], d["beta"]] for d in records]),
                    weights=np.array([d["weights"] for d in records]),
                    target_rhos=np.array([d["target_rho"] for d in records]),
                    reconstructed=np.array([d["reconstructed"] for d in records]),
                    frobs=np.array([d["frob"] for d in records]),
                    kets=pool_kets_dict, projectors=atoms,
                    source_labels=source_labels, occ_basis=np.array(EVEN_OCC))

r_star = solve_signed_lp(rho_star, atoms)
recon_star = np.tensordot(r_star["weights"], atoms, axes=(0, 0))
np.savez_compressed(os.path.join(OUT, "hubbard_star.npz"),
                    rho_star=rho_star, Ww=Ww,
                    kets=np.array(pool_kets_dict),
                    projectors=atoms, source_labels=source_labels,
                    occ_basis=np.array(EVEN_OCC),
                    weights=r_star["weights"],
                    target_rho=rho_star, reconstructed=recon_star,
                    star_frob=float(np.linalg.norm(recon_star-rho_star)),
                    witness_ratio_pool=ratio, witness_sup_pool=sup_pool,
                    witness_sup_grid=sup_grid, witness_opnorm=opnorm,
                    w0=w0, wD=wD, wh=wh)

summary = {
    "grid": rows,
    "u0_baseline_Rnorm": u0_baseline,
    "grand_canonical": True,
    "beta_0p5_not_maxmixed": True,
    "slater_atoms_added": int((source_labels == "free_singleparticle_wedge").sum()),
    "witness": {"U": U_star, "beta": beta_star, "w0": w0, "wD": wD, "wh": wh,
                "Tr_W_rho": num, "sup_pool": sup_pool, "sup_grid": sup_grid,
                "opnorm_bound": opnorm,
                "ratio_pool": ratio, "ratio_continuous_bound": num / sup_grid,
                "ratio_rigorous_opnorm": num / opnorm,
                "pool_size": len(pool_kets),
                "pool_unstable": ("ratio shrinks as pool refines; finite-pool number "
                                  "is NOT a reproducible certificate (review C-F6)"),
                "witness_valid_finite_pool": bool(num / sup_pool > 1),
                "witness_valid_rigorous": bool(num / opnorm > 1)},
    "walltime_s": time.time() - t_start, "seed": SEED,
}
with open(os.path.join(OUT, "hubbard_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] walltime {time.time()-t_start:.2f}s")
