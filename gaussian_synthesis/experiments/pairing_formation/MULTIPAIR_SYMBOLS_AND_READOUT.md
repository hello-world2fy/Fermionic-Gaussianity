# Two-transfer Gaussian symbols and phase-cycled counting readout

The purpose is to connect a full-Gaussian configuration coordinate to a
physical discrepancy in pair-transfer noise. Neither a higher polynomial
degree nor a successful symbol identity establishes new physics.
The predictive pilot and its negative controls are reported separately in
[REPORT_MULTIPAIR_NOISE.md](REPORT_MULTIPAIR_NOISE.md).

## Exterior symbols at second Cartan order

Let there be \(n\) physical complex fermion modes, and write
\(S=P_2(u\otimes u)\), \(C=S\bar u\), and \(B_u(v)=S\bar v\).
For distinct Majorana indices \(a_1,\ldots,a_{2m}\), define
\(O=i^m\gamma_{a_1}\cdots\gamma_{a_{2m}}\) and the Pfaffian polynomial
in the total quadratic generators

\[
W_m=\operatorname{Pf}\left[i\gamma_{a_j}^{(1)}\gamma_{a_k}^{(1)}
+i\gamma_{a_j}^{(2)}\gamma_{a_k}^{(2)}\right]_{j,k=1}^{2m}.
\]

On \(k\) identical Gaussian copies its covariant symbol is
\(\prod_{j=0}^{m-1}(k+2j)\langle O\rangle_g\).
The exterior component has the Berezin eigenvalue

\[
b_{(1^m)}(k)=\prod_{j=0}^{m-1}
\frac{k+2j}{k+2n-2-2j}.
\]

This specializes established compact symmetric-space Berezin calculus,
not a new quantization principle; see
[Zhang, Theorem 3.6, author-uploaded text](https://www.researchgate.net/publication/225423870_Berezin_transform_oncompact_Hermitian_symmetric_spaces)
and the [publisher record](https://link.springer.com/article/10.1007/s002290050109).
For the independent exterior components one can take \(m\le n/2\);
higher Majorana weights in a fixed parity representation reduce by
Clifford duality. The physical two-transfer application has \(n=28,m=4\).
At \(k=2\), the insertion on the Cartan range is

\[
\mathcal T_2(\langle O\rangle_g)=W_m/D_m,
\qquad D_m=\prod_{j=0}^{m-1}(2n-2j).
\]

The Pfaffian can be evaluated without enumerating operator matrix
products. Partition the ordered indices into an even subset \(A\) and
its complement \(A^c\), and let \(\varepsilon(A,A^c)\) be the shuffle
sign. All internal matchings on each copy collapse to the same Clifford
word, giving

\[
W_m=i^m\sum_{|A|\text{ even}}\varepsilon(A,A^c)
(|A|-1)!!(2m-|A|-1)!!\,
\gamma_A\otimes\gamma_{A^c},\qquad(-1)!!=1.
\]

The multiplicities count matchings within each subset; the matching
signs cancel the internal Clifford permutation signs. A tensor term
\(X\otimes Y\) contracts as \(X S\overline{Y^\dagger u}\).
Complex multilinearity extends the formula to a product of mutually
anticommuting creation/annihilation factors on distinct physical modes.
It does not apply without extra contractions when those modes repeat.

For
\[
O=s_{p_1}^+s_{p_2}^+s_{h_1}^-s_{h_2}^-,
\]
all four levels must be distinct. There are 128 even subsets, with
\(D_4=(2n)(2n-2)(2n-4)(2n-6)\). The label is the complete Gaussian
eight-fermion Wick/Pfaffian expectation, not merely a product of onsite
anomalous fields. The coherent output is

\[
M_O=d_2\int\langle O\rangle_g
|\langle g|u\rangle|^2\langle g|u\rangle g\,d\mu(g).
\]

## Intermediate sectors and a removable background

The coproduct terms visit intact-pair numbers \(r-2,\ldots,r+2\), with
zero, two, or four singly occupied levels on each copy. Let \(b\) be the
number of such levels. Their spin patterns are complementary across
copies. For an initial paired-copy difference on \(s\) levels, the
matrix coefficient is

\[
(-1)^{b/2+|I\cap B|+d_\downarrow}
\frac{\mathrm B(d+1/2,s-d+1/2)}{\pi},\qquad
d=|(P\triangle I)\setminus B|+b/2.
\]

\(I\) is the initial first-copy paired subset within the differing
levels, \(P\) its intact-pair output subset, \(B\) the broken levels,
and \(d_\downarrow\) the number of down singles in the first output.
This follows from the full-Fock zero-Fourier Cartan kernel and its
alternating Clifford sign. The implementation groups identical
union/intersection patterns and never replaces the full projector by a
projector onto the final paired space.

The two endpoint terms are

\[
M_O^{\rm end}=\frac{105}{D_4}
\left[O C+B_u(O^\dagger u)\right].
\]

Their subtraction leaves a cross-copy coherent contribution. It
vanishes for a paired occupation-basis Slater witness: then
\(S=u\otimes u\), and every nonempty proper ladder word has zero
expectation. In particular \(M_O=105Ou/D_4\) there.
The subtraction is **not** a positive configuration bin and is **not**
the connected target cumulant defined below. It is a decomposition of
the coherent operator insertion, used as a candidate coordinate.

Independent checks include Clifford duality of six/eight-Majorana
words against lower-degree insertions, full-Fock intermediate matrices,
the prior four-fermion formula, all paired Slater basis states at four
levels, and a separate full-SO Haar/Wick integral. The dense backend
remains exponential and is limited to \(L\le14\).

## Transfer noise and an interaction pulse

Let
\[
X_q=\sum_{ij}\frac{\cos[2\pi q(i-j)/r]}r
s_{p,i}^+s_{\ell,j}^-,\quad [\hat m,X_q]=X_q.
\]
The physical phase-sensitive noise is
\[
\mathcal C_{qp}=\langle X_qX_p\rangle
-\langle X_q\rangle\langle X_p\rangle.
\]
For the Hermitian quadratures \(Q_+=X+X^\dagger\) and
\(Q_-=-i(X-X^\dagger)\),
\[
\operatorname{Var}(Q_+)-\operatorname{Var}(Q_-)
=4\operatorname{Re}\mathcal C_{qq}.
\]
For a normalized pure state, the survival loss under
\(e^{-i\epsilon Q}\) equals \(\epsilon^2\operatorname{Var}(Q)+O(\epsilon^4)\).
These are ordinary quantum identities. The pulse uses a quartic
pair-exchange Hamiltonian; it is not a Gaussian unitary.

## Identical counting and four-fermion data, different readout

For a fixed-total-number paired density matrix \(\rho\), compare
\[
\rho_{\rm even}=\tfrac12(\rho+e^{-i\pi\hat m}\rho e^{i\pi\hat m}),
\qquad
\rho_{\rm diag}=\sum_{m=0}^r\Pi_m\rho\Pi_m.
\]
The second channel can also be implemented as an average over \(r+1\)
equally spaced count phases. These phase rotations are quadratic
fermion unitaries. Both channels preserve the entire occupation-basis
diagonal and every \(\Delta m=0\) matrix block, and both remove odd
count coherence.

Within the paired fixed-number sector, every nonzero four-fermion
expectation is a density product or a one-pair transfer. Its count
change is at most one. Therefore the two states have identical FCS
and identical **all** four-fermion expectations. They are not identical
states: the first channel retains \(\Delta m=\pm2,\pm4,\ldots\).

Apply
\[
U_{\epsilon,\phi}=e^{-i\epsilon Q_\phi},\qquad
Q_\phi=e^{-i\phi}X+e^{i\phi}X^\dagger.
\]
Define the difference as even-twirled minus fully count-dephased. Then
\[
\Delta\operatorname{Var}(m)=
-2\epsilon^2\operatorname{Re}\!\left[e^{-2i\phi}
\operatorname{Tr}(\rho X^2)\right]+O(\epsilon^4).
\]
To derive this, put \(R=[Q_\phi,m]=-e^{-i\phi}X+e^{i\phi}X^\dagger\).
The double commutator of \(m^2\) is
\([Q,R]m+m[Q,R]+2R^2\). The first two terms have count change zero and
cancel between the channels; the surviving part of \(R^2\) is
\(e^{-2i\phi}X^2+e^{2i\phi}X^{\dagger2}\).
The mean-count difference starts at fourth order, and parity removes
odd powers. Thus phases zero and \(\pi/4\) read the real and imaginary
parts of the surviving two-transfer coherence.

This is an application of established coherence-order selection and
phase cycling, related to multiple-quantum-coherence protocols such as
[Gärttner et al.](https://arxiv.org/abs/1608.08938).
It is not a new coherence principle, a new superconducting phase, an
implemented laboratory protocol, or a canonical-specific discovery.
The finite readout study uses already inspected conditions; it is a
diagnostic, not a new-parameter prediction.

## Stochastic controls require a Pauli-reduced measurement

For either Gaussian-seeded spin mean field or discrete TWA, let
\(p_i=s_{p,i}^x+i s_{p,i}^y\), \(h_j=s_{\ell,j}^x-i s_{\ell,j}^y\),
and \(X=\sum_{ij}W_{ij}s_{p,i}^+s_{\ell,j}^-\).
The operator identities \((s_i^\pm)^2=0\) must be applied before
factorizing the spin symbol. With
\(a_i=p_i\sum_jW_{ij}h_j\) and \(b_j=h_j\sum_iW_{ij}p_i\), this gives

\[
(X^2)_{\rm symbol}
=\left(\sum_i a_i\right)^2-\sum_i a_i^2-\sum_j b_j^2
+\sum_{ij}W_{ij}^2p_i^2h_j^2.
\]

This is inclusion-exclusion of repeated upper and lower sites. Every
remaining term acts on four distinct spin sites. Simply squaring the
classical transfer amplitude gives an incorrect symbol. An independent
test constructs the complete four-spin matrices and checks this expression
on random product states. It does not make the subsequent mean-field
propagation exact.

The Gaussian initialization samples transverse spin components with
variance \(1/4\); discrete TWA uses independent values \(\pm1/2\).
Both keep the initial longitudinal component fixed by the Fermi sea.
The Hamiltonian uses the exact onsite identity
\(s_i^+s_i^-=s_i^z+1/2\), hence excludes transverse self-fields.
For context, see [Lacroix, Gambacurta and Ayik](https://arxiv.org/abs/1303.0748)
on pairing fluctuations in stochastic mean field and
[Schachenmayer, Pikovski and Rey](https://arxiv.org/abs/1408.4441)
on discrete TWA. The random classical seeds are not draws from the
fermionic Gaussian canonical measure and do not define positive quantum
density matrices. We compare the weak-pulse coefficient through its
operator identity, not a putative pure-state survival probability of
these classical trajectories.

The angular controls use two independent seeds at each of three sizes,
with 131072 trajectories per seed and initialization. Two time steps,
0.004 and 0.002, reuse identical random samples to isolate step error.
There are 1572864 independent angular trajectories, plus 524288 for the
original uniform L12 G2 problem. Step repeats do not double these totals.
Reported block standard errors are estimates, not rigorous confidence
certificates. Both physical conditions were already inspected.
