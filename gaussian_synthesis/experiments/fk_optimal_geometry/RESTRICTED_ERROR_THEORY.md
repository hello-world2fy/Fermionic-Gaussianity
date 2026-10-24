# Energy windows, coefficient budgets and unavoidable physical errors

Let A be any phase-closed subset of normalized Gaussian rays, and let
`phi=sum c_j g_j` be normalized with `sum|c_j|<=B` and all g_j in A.
For any test vector w, a physical target overlap lower bound L, a restricted
support upper bound h, and a Hilbert norm upper bound n, Cauchy–Schwarz gives

`||psi-phi|| >= [L-B h]_+/n = d`.

Because coefficient cost and membership in A are unchanged by an overall
phase, the same bound applies after aligning phi with psi. Consequently
`||psi-phi_aligned||²=2-2|<psi,phi>|`, and for0<=d<=sqrt2,

`F=|<psi,phi>|² <= (1-d²/2)²`.

If d>sqrt2 the admissible normalized class is empty. If d=0 the bound is
uninformative, which must be reported rather than treated as a failed run.
The inequalities apply to arbitrary finite or convergent continuous
Gaussian decompositions and do not depend on the number of atoms.

For the FK window `E(g)<=E_G+Delta`, t>=9, the predecessor proves
`E_G=-14-4t`, `18x²+4(t-9)x<=Delta`, `eta>=(1-x)²`, where
`x=1+<H_hop>/4`. Thus x<=X, with

`X=2Delta/[sqrt(16(t-9)²+72Delta)+4(t-9)]`.

For X<1, the exact endpoint witness w0 has norm2 and restricted support
`h<=1-(1-X)²/2`. Substituting its certified physical L0 yields D1.
The restricted support itself is sharp on the known mass/hopping arc;
the Cauchy–Schwarz fidelity bound need not be sharp for a particular target.

For a unique ground state of the full256-dimensional Hamiltonian, a strict
gap lower bound gamma gives

`<phi,H phi>-E0 >= gamma(1-F) >= gamma(d²-d⁴/4)`.

Our spectral certificate uses directed interval congruences within all65
Fock connectivity blocks (largest dimension18). An independent exact
rational LDL calculation verifies the inertia at both ground-energy
endpoints and below the first excited level. This includes both parity
sectors; a gap in the three-dimensional singlet alone would be insufficient.

Delta and budget multipliers are rational decimals. The saved budget is an
explicit outward-rounded binary64 upper threshold, and every reported bound
uses that threshold. This makes the small conservative rounding choice
auditable. State overlaps include the physical eigenvector error enclosure.

These are general dual and spectral inequalities applied to explicit FK
support functions, not novelty claims about Cauchy–Schwarz, Markov or the
spectral gap inequality. Optimizing restricted Gaussian dictionaries and
testing whether the resulting bounds are sharp remain separate D2/E tasks.
