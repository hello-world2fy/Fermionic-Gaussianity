"""Quick numeric validation of the core math before the full experiment."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sggen.gaussian import (gaussian_ket_even, magic_ket, density_matrix,
                            sample_even_gaussian_rays, QUADRIC_PAIRS, EVEN_OCC)
from sggen.synthesis import solve_signed_lp

rng = np.random.default_rng(20261026)

# 1. pairing chart: normalization + quadric
Z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
Z = (Z - Z.T) / 2
g = gaussian_ket_even(Z)
print("norm of pairing Gaussian:", np.linalg.norm(g))
q = 0
for a, b, s in QUADRIC_PAIRS:
    q += s * g[a] * g[b]
print("quadric g^T J g (should be ~0):", 2 * q)

# 2. magic state M: not Gaussian
M = magic_ket()
qM = 0
for a, b, s in QUADRIC_PAIRS:
    qM += s * M[a] * M[b]
print("M quadric violation (should be nonzero):", 2 * qM)

# 3. analytic witness: sup over even g of |<M|g>|^2 == 1/2
#    product family g(t,t) saturates
overlaps = []
ts = np.linspace(0, 10, 2001)
for t in ts:
    a = 1.0 / (1.0 + t ** 2)
    ket = np.zeros(8, dtype=np.complex128)
    ket[0] = a; ket[1] = a * t; ket[6] = a * t; ket[7] = a * t ** 2
    overlaps.append(abs(np.vdot(M, ket)) ** 2)
print("max |<M|g(t,t)>|^2 on product family (expect 0.5):", max(overlaps))

# random even rays: max overlap squared
rays = sample_even_gaussian_rays(rng, 20000)
ov = np.abs(rays.conj() @ M) ** 2
print("max |<M|g>|^2 over 20000 random even rays (expect <=0.5):", ov.max())

# 4. LP for rho_M
rhoM = density_matrix(M)
atoms = [density_matrix(g)]
# add all 8 even Fock basis states (Slater determinants, Gaussian)
for i in range(8):
    e = np.zeros(8, dtype=np.complex128); e[i] = 1.0
    atoms.append(density_matrix(e))
# add product family atoms on grid
for t in np.linspace(0.05, 6.0, 40):
    a = 1.0 / (1.0 + t ** 2)
    ket = np.zeros(8, dtype=np.complex128)
    ket[0] = a; ket[1] = a * t; ket[6] = a * t; ket[7] = a * t ** 2
    atoms.append(density_matrix(ket))
# random atoms
for ket in sample_even_gaussian_rays(rng, 300):
    atoms.append(density_matrix(ket))
atoms = np.array(atoms)
print("dictionary size:", atoms.shape[0])

t0 = time.time()
res = solve_signed_lp(rhoM, atoms)
print(f"LP R(rho_M) = {res['value']:.6f}  residual={res['residual']:.2e}  "
      f"nonzero={res['n_nonzero']}  time={time.time()-t0:.2f}s  status={res['status']}")

# 5. witness value
W = 4 * rhoM - np.eye(8)
print("Tr(W rho_M) (expect 3):", np.trace(W @ rhoM).real)
