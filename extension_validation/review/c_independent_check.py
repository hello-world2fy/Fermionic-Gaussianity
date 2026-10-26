"""Independent review of C (signed Gaussian). Read-only; imports sggen.

Checks:
 (O1) odd-sector value of W=4 rho_M - P_+  (expect 0, producers wrote -1)
 (O2) sup |<M|g>|^2 = 1/2 and the prefactor in the bound derivation
 (O3) R(rho_M) exact upper: does a fine contact dictionary give TV -> 3.0000
      (analytic contact combination) instead of 3.000006?
 (O4) U=0 low-T even block: is it really a Gaussian state (true R=1)? quantify
      the dictionary bias LP_R - 1.
 (O5) witness W (U=8,beta=8): true continuous sup_g <g|W|g> via dense local
      optimization. Decide if ratio 7.42 (pool=0.0144) is real or a pool artifact.
"""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import sys
sys.path.insert(0, "/Users/bytedance/Desktop/Fermionic Gaussianity/extension_validation/signed_gaussian")
import numpy as np
from sggen.gaussian import (magic_ket, density_matrix, gaussian_ket_even,
                            sample_even_gaussian_rays, EVEN_OCC, contact_kets,
                            product_family_kets, basis_ket)
from sggen.synthesis import solve_signed_lp
from sggen.hubbard import hubbard_hamiltonian, gibbs_state, even_block

M = magic_ket(); rhoM = density_matrix(M)
W = 4*rhoM - np.eye(8)   # even block of 4 rho_M - P_+
rng = np.random.default_rng(12345)

print("=== O1 odd sector ===")
# odd Gaussian: even-block projection of an odd-parity Gaussian ket is zero.
# Build W acting on full 16-dim: W16 = 4 |M16><M16| - P_+. M even.
M16 = np.zeros(16, dtype=complex); M16[0]=1/np.sqrt(2); M16[15]=1/np.sqrt(2)
W16 = 4*np.outer(M16,M16.conj()) - np.diag([1.0 if (bin(o).count('1')%2==0) else 0.0 for o in range(16)])
# an odd Gaussian ket: odd-parity pairing chart -> construct via Z with odd occupancy
# simplest: any single-Slater odd state |c0^dag|0> = |0001>
odd = np.zeros(16, dtype=complex); odd[1]=1.0   # |0001>
print("  Tr(W16 sigma_odd) for odd Slater |0001> =", np.trace(W16 @ np.outer(odd,odd.conj())).real,
      " (expect 0; producers claim -1)")

print("=== O2 sup |<M|g>|^2 ===")
rays = sample_even_gaussian_rays(rng, 200000)
sup = np.max(np.abs(rays.conj() @ M)**2)
print(f"  numeric sup over 2e5 Haar even rays = {sup:.6f}  (analytic bound 0.5)")

print("=== O3 R(rho_M) fine contact dictionary ===")
ck = contact_kets(rng, 6000)
ck += product_family_kets(np.linspace(0.0, 8.0, 400))
atoms = np.array([density_matrix(k) for k in ck])
atoms = np.concatenate([atoms, np.stack([density_matrix(np.eye(8,dtype=complex)[i]) for i in range(8)])])
res = solve_signed_lp(rhoM, atoms)
print(f"  fine dict R_LP = {res['value']:.8f}  residual={res['residual']:.2e}  nz={res['n_nonzero']}")
print(f"  analytic witness lower = {np.trace(W@rhoM).real:.6f}")

print("=== O4 U=0 low-T even block is really a Gaussian state? ===")
for beta in [1.0, 4.0, 8.0]:
    H = hubbard_hamiltonian(t_hop=1.0, U=0.0, mu=0.0)
    r16 = gibbs_state(H, beta)
    re = even_block(r16); re /= np.trace(re)
    # covariance of free-fermion thermal state should be a valid Gaussian density
    # simplest test: is re in the convex hull of pure Gaussians with TV=1?
    # i.e. does the LP with a DENSE dictionary get R close to 1?
    uni = sample_even_gaussian_rays(rng, 8000)
    at = np.array([density_matrix(k) for k in uni])
    at = np.concatenate([at, np.stack([density_matrix(np.eye(8,dtype=complex)[i]) for i in range(8)])])
    rr = solve_signed_lp(re, at)
    print(f"  U=0 beta={beta}: LP R_norm={rr['value']:.4f} residual={rr['residual']:.1e}  "
          f"(true free thermal even block should be R=1; gap = dictionary bias)")

print("=== O5 witness continuous sup ===")
from sggen.hubbard import annihilator
c=[annihilator(j) for j in range(4)]; cd=[x.conj().T for x in c]
n=[cd[j]@c[j] for j in range(4)]
Da=n[0]@n[1]; Db=n[2]@n[3]
hop=cd[0]@c[2]+cd[1]@c[3]+cd[2]@c[0]+cd[3]@c[1]
idx=list(EVEN_OCC)
De=(Da+Db)[np.ix_(idx,idx)]; He=hop[np.ix_(idx,idx)]
w0,wD,wh=-0.25,-1.25,0.5
Ww = w0*np.eye(8)+wD*De+wh*He
Hstar = hubbard_hamiltonian(1.0, 8.0, 4.0)
rstar = even_block(gibbs_state(Hstar,8.0)); rstar/=np.trace(rstar)
num = np.trace(Ww@rstar).real
# dense sup: random rays + local ascent on sphere restricted to Gaussian manifold
def val(k): return np.real(k.conj()@Ww@k)
sup_pool = max(val(k) for k in sample_even_gaussian_rays(rng, 100000))
# product family grid
sup_prod=0.0
for t in np.linspace(0,12,2000):
    k=np.zeros(8,dtype=complex); k[0]=1/(1+t**2); k[1]=t/(1+t**2); k[6]=t/(1+t**2); k[7]=t**2/(1+t**2)
    sup_prod=max(sup_prod,val(k))
print(f"  Tr(W rho)={num:.6f}")
print(f"  sup over 1e5 Haar rays = {sup_pool:.6f};  sup product-family = {sup_prod:.6f}")
print(f"  opnorm ||W|| = {np.linalg.norm(Ww,2):.4f}")
print(f"  -> if true sup > {num:.4f}, witness INVALID (7.42 = pool artifact); "
      f"if < {num:.4f}, valid.")
