# Gaussian pair fields, normal exchange, and coherent symbol insertions

All atoms in this note are normalized pure Gaussian states in the full
even Fock space of \(n=2L\) complex physical modes. The witness and final
outputs may lie in a fixed-number paired sector. Quadratic intermediate
states must not be restricted to that sector.

## Physical coordinate split

Define \(\rho_{ab}=\langle c_a^\dagger c_b\rangle_g\),
\(\kappa_{ab}=\langle c_a c_b\rangle_g\), and
\(\Delta_i=\kappa_{i\downarrow,i\uparrow}=\langle s_i^-\rangle_g\).
Wick's identity gives

\[
f_{ij}(g)=\langle s_i^+s_j^-\rangle_g
=\underbrace{\Delta_i^*\Delta_j}_{a_{ij}(g)}
+\underbrace{\rho_{i\uparrow,j\uparrow}\rho_{i\downarrow,j\downarrow}
-\rho_{i\uparrow,j\downarrow}\rho_{i\downarrow,j\uparrow}}_{e_{ij}(g)}.
\]

The anomalous-field product \(a\) retains a relative pairing phase.
The normal term \(e\) contains interlevel single-fermion coherence.
For \(i=j\), it is the determinant of the local normal density matrix.
Both are labels on Gaussian configurations; their canonical-measure
marginals are not physical target expectation values.

For an onsite product of paired Gaussian factors, \(e_{ij}=0\) when
\(i\ne j\). General Gaussian atoms need not have this property even
when the reconstructed paired target has zero off-diagonal one-body
density. This makes the split a concrete candidate to test, but does
not by itself establish that those atoms or coordinates are necessary.

The coherent vector associated with a label \(f\) is
\[
M_f(u)=d_2\int f(g)|\langle g|u\rangle|^2\langle g|u\rangle\,g\,d\mu(g).
\]
The experiments compute \(M_a,M_e,M_f=M_a+M_e\) separately.
They do not interpret an intermediate broken-pair contraction as \(e\).

## Quartic and product insertions

Let \(Q_{ab}=i\gamma_a\gamma_b\), with distinct Majoranas, and let
\(T_{ab}=Q_{ab}^{(1)}+Q_{ab}^{(2)}\) on the second Cartan representation.
For four distinct \(a,b,c,d\), set
\[
O=-\gamma_a\gamma_b\gamma_c\gamma_d,\quad
W=T_{ab}T_{cd}-T_{ac}T_{bd}+T_{ad}T_{bc}.
\]
The Gaussian covariant symbol of \(W\) on \(k\) copies is
\(k(k+2)\langle O\rangle_g\). The exterior-four-form Berezin eigenvalue is
\[
b_{(1,1)}(k)=
\frac{k(k+2)}{(k+2n-2)(k+2n-4)}.
\]
Thus at \(k=2\),
\[
\mathcal T_2(\langle O\rangle)=\frac{W}{D_-},
\qquad D_-=2n(2n-2).
\]
The tensor-product and spectral input is the established compact
symmetric-space Berezin calculus, specialized here to
\(\mathrm{SO}(2n)/\mathrm U(n)\), with root parameter \(a=4\) and genus
\(p=2n-2\). See
[Zhang, Theorem 3.6 (1998)](https://www.researchgate.net/publication/225423870_Berezin_transform_oncompact_Hermitian_symmetric_spaces)
and the [publisher record](https://link.springer.com/article/10.1007/s002290050109).
No novelty claim is made for this representation-theoretic machinery.

Products of two quadratic symbols contain both an exterior-four-form
part and a traceless symmetric part. The latter has eigenvalue
\[
b_{(2)}(k)=\frac{k(k-1)}{(k+2n-2)(k+2n-1)}.
\]
For disjoint index pairs, put \(J=T_{ab}T_{cd}\). The product insertion is
\[
\mathcal T_2(\langle Q_{ab}\rangle\langle Q_{cd}\rangle)
=\frac{J+W/(2n-2)}{D_+},\qquad D_+=2n(2n+1).
\]
One way to see the decomposition is that \(J-W/3\) has symbol
\(2[\langle Q_{ab}\rangle\langle Q_{cd}\rangle-\langle O\rangle/3]\)
at \(k=2\). The two components therefore require different denominators.
Complex linearity extends the expression to pair creation/annihilation
quadratics. It does not permit dropping their number-changing action.

## Paired final outputs with full intermediate sectors

Write \(S=P_2(u\otimes u)\), \(C=S\bar u\), and
\(B_u(v)=S\bar v\). For \(i\ne j\), take \(O=s_i^+s_j^-\), and define
\[
R_{ij}=OC+B_u(O^\dagger u),\qquad
Y_{ij}=s_i^+B_u(s_j^+u)+s_j^-B_u(s_i^-u).
\]
The anomalous-field insertion then has the useful contracted form
\[
M_{a_{ij}}=
\frac{R_{ij}+Y_{ij}+2n M_{f_{ij}}}{D_+},\qquad
M_{e_{ij}}=M_{f_{ij}}-M_{a_{ij}}.
\]

The full quartic insertion has
\[
M_{f_{ij}}=\frac{3R_{ij}+Y_{ij}-4Z_{ij}}{D_-}.
\]
Here \(Z_{ij}\) contracts \(S\) between complementary singly occupied
patterns on levels \(i,j\), with \(r-1\) intact pairs elsewhere, and embeds
the result into outputs with level \(i\) paired and \(j\) empty.
It is a computational intermediate, **not** the normal physical label.

For the onsite case, \(q=2n_{i\uparrow}-1\) equals
\(2n_{i\downarrow}-1\) on paired amplitudes. Let
\[
M_q=\frac{qC+B_u(qu)}{2n},\qquad
M_4=\frac{6C+2qB_u(qu)+4Y_{ii}}{D_-}.
\]
Then
\[
M_{f_{ii}}=\frac{C+2M_q+M_4}{4},\qquad
M_{a_{ii}}=\frac{(4+2n)C+4Y_{ii}+2nM_4}{4D_+}.
\]
The onsite formula includes the local spin-sector information of the
full Gaussian dictionary; \(f_{ii}\) is double occupation in full Fock
space, not merely a single-mode occupation symbol.

## A contrast that vanishes on paired Slater witnesses

For distinct levels, the two insertion formulas imply
\[
M_{a_{ij}}-\tfrac13 M_{f_{ij}}
=\frac{2(Y_{ij}+2Z_{ij})}{3D_+}.
\]
The cancellation follows by substituting the expression for \(M_f\)
and using \(2n/D_+-1/3=-D_-/(3D_+)\).
For any paired occupation-basis Slater determinant \(u\),
\(P_2(u\otimes u)=u\otimes u\). Its two-copy matrix has neither
number-changing paired entries nor broken-pair entries. Consequently
\(Y=Z=0\), and this coherent contrast vanishes exactly.

In the same case,
\[
M_f=\frac{3O u}{D_-},\qquad M_a=\frac{O u}{D_-},\qquad
M_e=\frac{2O u}{D_-}.
\]
Thus the normal-to-anomalous ratio of two already occurs at a Gaussian
Fermi sea. It does not establish a physical normal-transport mechanism.
The contrast label itself is not pointwise zero on Gaussian atoms.
Only the stated coherent integral vanishes for the paired Slater
witnesses; no claim about arbitrary witnesses or observable marginals
follows from this calculation.

After inspecting the original full/split pilot, we also selected the
leading eight or sixteen offsite contrast directions from the same four
early training times. This is a retrospective diagnostic. It improves
some errors relative to the raw split, but the unoptimized contrast
performs similarly and ordinary Hamiltonian directions remain much
more accurate. Neither the identity nor the improved coordinate choice
establishes a canonical-specific physical mechanism.

## Kernel implementation and independent checks

On a two-copy Fock block with \(s\) physical modes differing between the
copies, the Majorana bilinear \(\sum_a\gamma_a\otimes\gamma_a\) becomes
a sum of signed commuting bit flips. Its zero Fourier coefficient is
\[
K_s(d)=
\begin{cases}
(-1)^{d/2}2^{-s}{s\choose s/2}
{s/2\choose d/2}/{s\choose d},&s,d\text{ even},\\
0,&\text{otherwise}.
\end{cases}
\]
An additional sign counts flipped positions with odd position in the
ordered list of differing modes. The small full-even-Fock backend uses
this expression directly and retains all particle-number and seniority
sectors.

For paired outputs, the resulting compressed kernel is the positive
Beta kernel. The large backend evaluates both the \(r,r\) and
\(r-1,r+1\) paired-copy matrices. For two fixed unpaired levels, it
evaluates the complementary spin-pattern matrix with its alternating
input sign. These are matrix elements of the full projector.

Independent tests cover:

- Projector dimension, idempotence and replica symmetry in full even
  Fock space, and agreement with the existing paired Beta kernel.
- Quartic insertions from a separately constructed third Cartan
  projector and its partial trace, through four physical modes.
- The anomalous-field product from an independently constructed fourth
  Cartan projector (rank 294 for four physical modes) and its two-copy
  partial trace.
- All sixteen level-pair correlators in a random four-pair-level target,
  compared with direct full-Fock Majorana contractions.
- Number-changing paired and broken-pair intermediate matrices,
  compared entry by entry with the full Fock reference.
- Vanishing of the offsite contrast on every paired Slater determinant
  in the four-level half-filled test sector.
- Independent full-SO Haar/Wick integration, including random
  non-Gaussian paired witnesses and a 28-mode Fermi-sea check.

These finite-algebra checks use floating point, and Monte Carlo standard
errors are estimates. They are not interval certificates.
The dense implementation remains exponential and is capped at \(L=14\).

## Frozen physical test and deletion

Use the four pre-existing early witnesses at \(t=.30,.60,.90,1.35\).
The same labels are also integrated with the direct physical states.
Sixteen representatives of intershell and intrashell pair correlations
are projected onto the exact cyclic/reflection/band-exchange invariant
output space. This is equivalent to symmetrizing the labels, not to
replacing a Gaussian atom by a Gaussian symmetry average.

The shared nine-dimensional space contains all endpoint count-sector
profiles and \(H\psi(1.35)\), so both exact Fermi and endpoint starts
and the endpoint Hamiltonian tangent are retained.
Anomalous, normal, and full-symbol residual columns are selected using
training-only SVD. The nested split adds eight normal-coordinate
directions to eight anomalous-coordinate directions.

Deleting exactly the added normal directions recovers the anomalous
space with the same initial state. Adding eight ordinary Hamiltonian
directions to that same anomalous space provides a stronger repair
control. Additional equal-dimensional full-symbol, unoptimized,
ordinary operator-history, and ordinary Krylov controls remain required.
None of these construction identities establishes a new physical
mechanism before the predictive and deletion comparisons.
