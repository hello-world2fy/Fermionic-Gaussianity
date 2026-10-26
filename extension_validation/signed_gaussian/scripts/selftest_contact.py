"""Diagnostic selftest (pure memory): contact-set dictionary for rho_M.

Verifies (a) contact atoms saturate |<M|g>|^2 = 1/2, and (b) the signed LP over the
contact dictionary reconstructs rho_M with a small FULL Frobenius residual.
Uses the current solve_signed_lp return fields (value/residual/frobenius_residual/
upper_valid).  Writes results to artifacts/logs/selftest_contact.log.
This is a diagnostic, not part of the main experiment pipeline.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sggen.gaussian import magic_ket, density_matrix, contact_kets, product_family_kets
from sggen.synthesis import solve_signed_lp, flatten_hermitian

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "logs")
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "selftest_contact.log"), "w")
def say(s):
    print(s); log.write(s + "\n")

rng = np.random.default_rng(20261026)
M = magic_ket(); rhoM = density_matrix(M)

kets = contact_kets(rng, 2000) + product_family_kets(np.linspace(0.02, 8.0, 60))
atoms = np.array([density_matrix(k) for k in kets])
for i in range(8):
    e = np.zeros(8, dtype=complex); e[i] = 1
    atoms = np.concatenate([atoms, density_matrix(e)[None]], axis=0)
say(f"dictionary size: {atoms.shape[0]}")

sat = [abs(np.vdot(M, k))**2 for k in kets[:8]]
say(f"sample |<M|g>|^2 of contact atoms (expect 0.5): {np.round(sat, 5).tolist()}")
assert all(abs(s - 0.5) < 1e-9 for s in sat), "contact saturation failed"

t0 = time.time()
res = solve_signed_lp(rhoM, atoms)
if res.get("status") != "ok":
    say(f"LP FAILED: {res.get('message')}")
    sys.exit(1)
w = res["weights"]
recon = np.tensordot(w, atoms, axes=(0, 0))
frob = float(np.linalg.norm(recon - rhoM))
say(f"LP R(rho_M) = {res['value']:.6f}  flatten_resid={res['residual']:.2e}  "
    f"Frobenius={frob:.2e}  upper_valid(<1e-8)={frob < 1e-8}  "
    f"nonzero={res['n_nonzero']}  time={time.time()-t0:.2f}s")
nz = np.where(np.abs(w) > 1e-9)[0]
say(f"nonzero atoms: {len(nz)}  sum|w|={np.abs(w[nz]).sum():.4f}  "
    f"positive={w[w>1e-9].sum():.3f} negative={-w[w<-1e-9].sum():.3f}")
say(f"RESULT: frobenius_ok={frob < 1e-8}  R={res['value']:.6f}")
log.close()
print("wrote", os.path.join(OUT, "selftest_contact.log"))
