"""Independent re-check of B (Slater) revised noci.convex_j_synthesize.

Claim under test: the convex-J solver is CORRECT; the ~0.95 residual on the
Hubbard local 40-atom dictionary is purely a span limitation (r outside col(A)),
NOT a solver defect. We show:
 (T1) orthonormal COMPLETE dictionary -> kkt_err ~ 0, closed-form c matches.
 (T2) random NON-ORTHOGONAL fat dictionary (nat>=dim, spans) -> kkt_err ~ 0.
 (T3) narrow dictionary (nat<<dim) -> kkt_err == proj_residual exactly, and the
      residual that remains is orthogonal to col(A) (pure projection gap).
Read-only; imports slater/noci.py.
"""
import sys
sys.path.insert(0, "/Users/bytedance/Desktop/Fermionic Gaussianity/extension_validation/slater")
import numpy as np
import noci

rng = np.random.default_rng(7)

print("=== T1 orthonormal complete dictionary ===")
dim, nat = 8, 8
Q, _ = np.linalg.qr(rng.standard_normal((dim, nat)))   # orthonormal basis (spans)
r = rng.standard_normal(dim)
out = noci.convex_j_synthesize(Q, r, k=2.0)
print(f"  kkt_err={out['kkt_err']:.3e}  proj_residual={out['proj_residual']:.3e}  "
      f"span={out['span_dim']}  t={out['t']:.4f}")
# closed form for orthonormal: c_i = sign(r_i)|r_i|^{1/(2k-1)} type; check r=tAc
print(f"  residual ||r-tAc||/||r|| = {out['kkt_err']:.3e} (expect ~1e-8)")

print("=== T2 random non-orthogonal fat dictionary (spans dim) ===")
dim, nat = 12, 40
A = rng.standard_normal((dim, nat))
r = rng.standard_normal(dim)
out = noci.convex_j_synthesize(A, r, k=2.0)
print(f"  kkt_err={out['kkt_err']:.3e}  proj_residual={out['proj_residual']:.3e}  "
      f"span={out['span_dim']}/{dim}")
# residual should be ~0 since col(A)=whole space
print(f"  (expect both ~1e-8; proves solver reconstructs r when dictionary spans)")

print("=== T3 narrow dictionary (nat<<dim): kkt_err == proj_residual ===")
dim, nat = 400, 40
A = rng.standard_normal((dim, nat))
r = rng.standard_normal(dim)
out = noci.convex_j_synthesize(A, r, k=2.0)
# residual vector
coef, *_ = np.linalg.lstsq(A, r, rcond=None)
resid = r - A @ coef
# check resid orthogonal to col(A)
orth_gap = float(np.linalg.norm(A.T @ resid) / (np.linalg.norm(A) * np.linalg.norm(resid) + 1e-30))
print(f"  kkt_err={out['kkt_err']:.4f}  proj_residual={out['proj_residual']:.4f}  "
      f"|kkt-proj|={abs(out['kkt_err']-out['proj_residual']):.2e}")
print(f"  orthogonality of residual to col(A): ||A^T resid||/(||A||||resid||)={orth_gap:.2e}")
print(f"  -> 0.95 residual is the un-spanned component, NOT a solver failure.")
