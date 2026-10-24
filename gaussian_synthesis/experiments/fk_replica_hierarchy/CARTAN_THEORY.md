# Exact low-order Cartan moments and reconstruction

This backend fixes the independent FK order-four reconstruction gate. It is
restricted to k=2,3,4; high orders still use tempered integration. The exact
integer result and its directed interval evaluation are separate from a
floating Chebyshev calculation and from the half-integer mass estimates.

## Quasiparticle coordinates

Let V have columns the hopping eigenvectors in the established singlet basis:

```
V = [[1/4, sqrt(14)/4, 1/4],
     [-sqrt(2)/2, 0, sqrt(2)/2],
     [sqrt(7)/4, -sqrt(2)/4, sqrt(7)/4]].
```

The Gaussian vacuum is S V[:,0]. With creation operators
`d_j†=(gamma_A,j-i gamma_B,j)/2`, recursive application constructs a full
unitary Fock basis. In that basis the three singlets are vacuum, a normalized
Cayley four-form, and minus the filled state. The four-form has masks
`15,51,60,85,90,102,105,150,153,165,170,195,204,240` and signs
`-,-,-,-,+,+,+,-,-,-,+,+,+,+`, divided by sqrt(14).
The implementation verifies this convention against physical Majoranas.
`exact_hopping_fock_proof` additionally verifies the change of basis entirely
in signed integers: all Fock vectors have denominator 4, every creation-step
division is exact, and the vacuum annihilation, full unitarity and Cayley
column identities hold as integer equalities.

These 16 supported bitstrings form the extended binary Hamming code. Conserved
per-mode total occupations permit restriction to only weight sectors reached
by their k-fold products. The resulting symmetric basis dimensions for k=2,3,4
are 528, 19184, 618448 (unrestricted: 8256, 357760, 11716640).

## Casimir and exact projector

On k even spinor copies put A=sum_(i<j) Lambda_ij², with
Lambda_ij=sum_a gamma_a^(i) gamma_a^(j). With the long-root squared length 2,
the total quadratic Casimir satisfies

`A=4(Cmax-Ctotal)`, `Cmax=2 k²+28 k`.

The zero space in the symmetric power is the unique highest-weight Cartan
component of weight `(k/2,...,k/2)`, containing every coherent tensor g^k.
Its orthogonal projector is therefore the coherent-state moment projector.
Uniqueness also follows directly from highest-weight theory: all component
weights are below the tensor highest weight; every nonzero dominant lowering
strictly lowers the Casimir. The chirality-reversed weight with the same
Casimir cannot occur: saturation of the first seven spinor coordinates fixes
the last coordinate in each same-chirality factor.

Every possible dominant constituent weight has components bounded by k/2.
For k=2 they have b entries 1 and the rest 0, b even; for k=3 they have m
entries 3/2 and the rest 1/2; for k=4 they have a entries 2, b entries 1 and
the rest 0, b even. The last sign may reverse without changing its Casimir,
since rho_8=0. This gives a superset of all nonzero defects D. Extra roots for
absent representations do not alter the projector:

`P = product_(d in D) (d I-A) / product_(d in D) d`.

In pair Fock coordinates x,y, let D=x XOR y contain m bits. The diagonal
Lambda² contribution is 4m. Flipping mismatch positions j,l in both copies
has coefficient `8 (-1)^(j+l)`, where j,l are ranks in the ordered mismatch
list. This action is independently checked at k=2 against the full physical
Majorana projector, including complex probes and derivatives.

Sorted k-tuples have multiplicity m_t=k!/prod_a n_a!. In ordered-amplitude
coordinates the inner product is diag(m_t), and A has integer entries. Its
weighted self-adjointness is checked exactly. The normalized-tensor version
is related by the diagonal similarity sqrt(m_t).

## Integer Gram and modular recovery

Use unnormalized four-form coordinates x=(beta_0,beta_1/sqrt(14),beta_2),
beta=V^T z. Let Y map monomials x^alpha into ordered symmetric amplitudes;
its nonzero entries are the products of the Cayley/filled-state signs.
The exact Gram numerator is

`N = Y^T diag(m_t) product_d(d I-A) Y`, denominator `D=product_d d`.

Thus q_k(z)=sum_(alpha,beta) conjugate(x^alpha) N_ab/D x^beta.
Every coefficient is an integer divided by D before the displayed radical
coordinate transform. This is not rational fitting of floating data.

Because P is an orthogonal contraction, each numerator obeys the a priori
bound `|N_ab| <= D max_alpha [multinomial(k,alpha) 14^alpha_1]`.
Projection is evaluated modulo distinct verified 31-bit primes. Chinese
remainders recover every signed integer uniquely once the combined modulus
exceeds twice this bound. Row-sum bounds prevent int64 overflow in every
sparse product and final Gram contraction. A separate replay checks all
stored residues, source hashes and the additional identity `A P Y=0` modulo
every prime. Order four needs six primes and has a 47-digit denominator.

An independent normalized Chebyshev filter of the same Casimir converges to
the projector. It saves all supported projected tensor rows and contracts
them back in all 128 Fock directions. Unsupported reconstructed directions
vanish by the conserved Hamming-code character; the 16 supported directions
are contracted without inserting a singlet projection. This is an additional
floating cross-check, not the exact arithmetic certificate itself.

## Interval certificates

`fk4/exact_cartan.py` evaluates the recovered integer polynomial, its complex
gradient and all radical basis changes with 70-digit directed intervals.
The gradient is C_k(z)=partial_(conj z) q_k/k. Spin(7) equivariance proves that
the complete C_k(Sz) lies in S; applying the exact radical S then encloses
the entire 128-component reconstruction vector and its residual.

A safely rounded witness is rechecked to satisfy q_k(w)<=1. The coherent
projector gives h_G(w)^(2k)<=q_k(w), so this certifies feasibility on the
complete Gaussian dictionary. The reported L applies to the exact vector of
saved binary64 target entries. It does not silently certify the exact
Hamiltonian ground state; that would require a separate eigenstate enclosure.

Adjacent integer moments provide a deterministic cost envelope by Cauchy:

`B_k <= sqrt((d_k/d_(k-1)) q_k(u) q_(k-1)(u))`.

The exact target residual can be repaired in the occupation basis with cost
at most its interval l1 bound. This yields a constructive, certified upper
bound even when the binary64 Newton solution is not the exact minimizer.

## Independent coefficient-mass estimate

The beta=3.5 samples were generated independently of the newly computed exact
canonical witness. For old/new witnesses define
`R=(|<g|u_new>|/|<g|u_old>|)^7`. Then

`B_4(new)=q_4(new) E_old[R]/E_old[|<g|u_new>| R]`.

This removes the old forward normalizer from the mass estimate. Saved full
Gaussian kets, new normalized probability weights and aligned phases define
the reweighted P_4. Five independent SMC repetitions supply empirical
Student-t intervals; ratio bias and finite-chain bias are not covered by a
strict probability guarantee. The certified Cauchy envelope is reported
separately. Monte Carlo reconstruction fluctuations are diagnostics, while
the canonical witness reconstruction has its independent interval bound.

## Transfer to the exact physical ground state

The separate `fk4/state_enclosure.py` closes the saved-target qualification.
It uses the exact integer interaction and half-integer imaginary hopping
matrices, with t defined by its supplied binary64 value. A diagonal phase
change makes the full 256-dimensional Hamiltonian real. Floating eigenvectors
Q are used only as a congruence matrix: a conservative error bound proves Q
invertible and verifies that `Q^T(H-bI)Q` has exactly one negative direction.
Strict signed diagonal dominance proves this inertia, so every excited
eigenvalue is above b. This includes the odd-parity sector.

The calculation assumes standard IEEE-754 dot-product rounding with no
overflow/underflow. With n=256, |Qij|<=2 and a power-of-two M enclosing H-bI,
the entrywise bound used for two products is `32 gamma_(2n) n² M`, where
gamma_m=m epsilon/(1-m epsilon) and epsilon=2^-52 (twice unit roundoff).
The bound includes the diagonal subtraction; orthogonality uses the analogous
one-product bound. Its substantial slack also covers evaluation of the
displayed bounds and row sums. This is a documented rounding-error argument,
not an unqualified claim that a floating eigensolve is certified.

Directed interval arithmetic then encloses the saved state's norm, Rayleigh
energy E and residual r. For the ground-state phase aligned with the saved
vector, its distance from the exact ground state is at most
`|norm(target)-1|+sqrt(2)||r||/(b-E)`.
Subtracting witness_norm times this distance transfers L to the exact ground
state. Adding sqrt(128) times this distance transfers the repaired Fock cost
upper bound. Across the nine order-four targets the distance bounds are below
1.4e-14. The original saved-target certificates remain separately available.
