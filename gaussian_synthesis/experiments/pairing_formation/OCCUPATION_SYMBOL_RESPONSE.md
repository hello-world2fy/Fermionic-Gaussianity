# Actual Gaussian occupation coordinates and density response

This note concerns the full even pure Gaussian orbit on \(n=2L\) physical
complex fermionic modes. An atom is normalized in the full Fock space.
The coordinate \(p_i(g)=\langle g|n_i|g\rangle\) is taken **before** number
or seniority projection. It is not the projected-atom coordinate in the
earlier count-bin experiments and is not an observable of the synthesized
target after taking a distribution marginal.

## Single-mode insertion at replica order two

Write \(P_2=d_2\int |g,g\rangle\langle g,g|d\mu(g)\), with
\(d_2=\frac12{2n\choose n}\), and let
\(N_i=n_i\otimes I+I\otimes n_i\).
For every integer \(d\ge0\),

\[
 d_2\int p_i(g)^d |g,g\rangle\langle g,g|d\mu(g)
 =\sum_{j=0}^2 \frac{(n-1+j)_d}{(2n)_d}\,P_{2,j},
 \qquad P_{2,j}=P_2\,1_{\{N_i=j\}} .
\]

Here \((a)_d\) is the rising factorial. The insertion is diagonal in the
**copy occupation** \(j\), not in a physical target occupation probability.

A derivation for this specific \(k=2\) identity is as follows.

1. For a Haar complex structure, one fixed off-diagonal covariance entry
   is a coordinate of a uniform sphere in \(2n-1\) real dimensions.
   Consequently \(p_i\) has the Beta\((n-1,n-1)\) law.
2. The second Cartan representation is a half of the middle exterior
   power of the \(2n\)-dimensional vector representation. Under
   \(\mathrm{Spin}(2)\times\mathrm{Spin}(2n-2)\), its \(N_i=0,1,2\)
   sectors are irreducible, with dimensions
   \(\frac12{2n-2\choose n-1},{2n-2\choose n-2},
   \frac12{2n-2\choose n-1}\), respectively. The insertion commutes with
   this subgroup, so Schur's lemma makes it scalar on each sector.
3. The squared norm of the \(j\)-component of \(|g,g\rangle\) is
   \({2\choose j}p_i^j(1-p_i)^{2-j}\). Taking a trace in that sector and
   dividing by its \(d=0\) value gives the Beta integral ratio above.

This is an application of coherent-state/Berezin quantization, not a
novel physical mode or a claim of priority. General compact symmetric
space Berezin spectra are established literature:
[Zhang, manuscripta mathematica 97, 371–388 (1998)](https://link.springer.com/article/10.1007/s002290050109).
That reference supplies the general setting; the particular normalization
and occupation contraction used here are derived explicitly above and
checked independently in the accompanying code.

## Contraction without Haar vector noise

Set \(a=\langle g|u\rangle\), and

\[
C=C_2(u)=d_2\int |a|^2a\,g\,d\mu,\quad
A_u(v)=d_2\int |a|^2\langle g|v\rangle g\,d\mu,\quad
B_u(v)=d_2\int a^2\langle g|v\rangle^*g\,d\mu.
\]

For the physical projector \(P=n_i\), put \(B=B_u(Pu)\). Contracting the
three copy-occupation sectors yields

\[
X_2=PB,\qquad X_1=PC+B-2PB,\qquad X_0=C-PC-B+PB.
\]

Therefore the actual coordinate-weighted coherent vector is

\[
M_{i,d}=d_2\int p_i(g)^d|a|^2a\,g\,d\mu
=\sum_{j=0}^2\frac{(n-1+j)_d}{(2n)_d}X_j .
\]

Gaussian-unitary equivariance separately implies the Ward identity
\(PC=2A_u(Pu)-B_u(Pu)\). In particular,

\[
M_{i,1}=\frac{n-1}{2n}C+\frac1n A_u(Pu),\qquad
M_{i,p-1/2}=\frac1n A_u((P-\tfrac12)u).
\]

The formula only uses the full-Gaussian Cartan map and its derivatives.
Its implementation compresses the **output and input states** to the
fixed-number paired sector; the Gaussian dictionary is unrestricted.
The calculated numbers are float64 tensor contractions, not interval
certificates. These bounded physical labels differ from the complex
overlap-ratio labels used in the previous A/B experiments.

There is a consequential limitation: all scalar functions of one
occupation coordinate generate vectors in the span of \(X_0,X_1,X_2\).
Increasing its polynomial degree cannot create arbitrarily many new
directions at fixed replica order two. This does not restrict joint
coordinates of different modes, pair-field phases, or higher replica orders.

For comparison, the linear frame satisfies
\[
d_1\int(p_i-\tfrac12)\langle g|\psi\rangle g\,d\mu
=\frac{(P-\tfrac12)\psi}{2n-1}.
\]
The centered square is a scalar multiple of \(\psi\):
\(d_1\int(p_i-\frac12)^2\langle g|\psi\rangle g\,d\mu
=\psi/[4(2n-1)]\).
Thus a zero-sum angular harmonic of that square contributes no extra
linear-frame direction.

## Physical response experiment

For the \(L=14\), equal-channel angular-patch model, use
\[
f_q(j)=\sqrt{2/r}\cos(2\pi qj/r),\quad
O_{q,\sigma}=\sum_j f_q(j)(n_{p,j}-\sigma n_{\ell,j}),\quad
q=1,2,3,\quad \sigma=\pm1 .
\]
Here \(n\) is one spin occupation; it equals pair occupation on the target
sector. The lower-shell hole occupation is \(1-n_{\ell,j}\).
The zero sum of \(f_q\) makes the physical operator equivalent to the
particle-plus/minus-hole density harmonic. These six response sectors lie
outside the old fully invariant 170-dimensional sector.

Freeze the centered Gaussian coordinate vectors
\[
\sum_jf_q(j)[M_{p,j,p-1/2}-\sigma M_{\ell,j,p-1/2}],
\quad
\sum_jf_q(j)[M_{p,j,(p-1/2)^2}+\sigma M_{\ell,j,(p-1/2)^2}].
\]
The changed sign in the second expression is necessary because a hole
map reverses a centered occupation but preserves its square.
Use either the endpoint \(t=1.35\) or the four already available training
times \(0.30,0.60,0.90,1.35\). No later states enter this selection.

Every response space explicitly contains the identical initial
\(v=O\psi(1.35)/\|O\psi(1.35)\|\). A common 32-vector ordinary Hamiltonian
Krylov space propagates the background. The predicted four-fermion
correlator is
\[
C_O(\tau)=\langle\psi(1.35+\tau)|Oe^{-iH\tau}O|\psi(1.35)\rangle.
\]
Its imaginary part gives the response to a weak Gaussian phase kick:
\[
\langle O\rangle_{e^{-iH\tau}e^{-i\alpha O}\psi}
-\langle O\rangle_{e^{-iH\tau}\psi}
=2\alpha\,\mathrm{Im}\,C_O(\tau)+O(\alpha^2).
\]
Its real part is the symmetrized density correlation. The initial mean
of each \(O\) vanishes by symmetry, so it is also the connected correlator.

The response-state count distribution belongs to the normalized state
created by \(O\). It is a diagnostic of that response ket; it is **not**
the transferred-pair distribution after a finite kick. The latter
requires the full perturbed wavefunction.

Controls include unoptimized cubic moments, ordinary early-time
\(O\psi\) and \(H O\psi\), equal-dimensional Hamiltonian Krylov spaces,
and deleting all added directions while retaining the same \(v\).
This isolates response structure; it does not establish a computational
advantage because a high-quality common background and full paired-space
matrix elements are supplied to all models.

Pump-induced collective pairing responses have substantial prior work,
including [Müller et al. (2019)](https://arxiv.org/abs/1907.10464) and
[Müller and Eremin (2021)](https://arxiv.org/abs/2107.02834).
The present finite-patch density response is not identified as a
Bardasis–Schrieffer mode, a new phase, or a thermodynamic excitation.
