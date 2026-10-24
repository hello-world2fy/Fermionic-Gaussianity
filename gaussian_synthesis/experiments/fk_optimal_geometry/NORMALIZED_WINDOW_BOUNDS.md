# Stronger physical bounds from normalized excitation constraints

These bounds apply to every normalized approximate state with coefficient
cost <=B and every actual Gaussian atom in E(g)<=E_G+Delta. They impose no
rank limit. They supplement, rather than invalidate, the earlier endpoint
witness distance bound. Budgets are the explicit reference U,1.1U,1.5U,2U
values from initial_certificates_02, including the original t20 upper bound.

## Global amplitude caps

Use the hopping Gaussian as quasiparticle vacuum. H_hop=N-4, and the global
window theorem gives <N>=4x with x<=X_Delta. For an even Gaussian, its four
canonical pair probabilities have sum p_i=2x. The singlet hopping basis has
occupation numbers 0,4,8. Thus

    |<e8,g>| <= X²/4,
    |<e4,g>| <= sqrt(3/2) X.

The first bound is AM–GM on four pair probabilities; the second follows from
the second elementary symmetric polynomial. Odd Gaussian states have zero
overlap with these even vectors and obey the same bounds trivially.

There is a stronger model-specific bound. The four-particle singlet e4 is the
normalized Cayley four-form. In the exact hopping Fock basis write the 28
Thouless pair amplitudes as z=(Z_ab)_(a<b). The exact integer pairing matrix T
obeys (T†T-I)(T†T-9I)=0 and tr(T†T)=84, so ||T||=3. Hence

    <e4,g> = a0 z.T T z/(2 sqrt14),
    |<e4,g>| <= 3 |a0| ||z||²/(2 sqrt14).

For X<1/2 the even Gaussian's vacuum overlap is nonzero: a0²=prod(1-p_i)
>=1-sum p_i>0. Also ||z||²=sum p_i/(1-p_i). For each i,
|a0|/(1-p_i)<=1/sqrt(1-sum p_i). Therefore

    |<e4,g>| <= 3 X/(sqrt14 sqrt(1-2X)).

The implementation uses the smaller of these two four-particle bounds.
The exact hopping Fock transformation, 14 integer Cayley coefficients, and
pairing-matrix polynomial are checked using dyadic/Gaussian-integer algebra.
The operator-norm statement holds for complex pairing coordinates too.

For any admissible superposition phi, put

    b4=B h4(X), b8=B X²/4.

The triangle inequality then bounds its two singlet amplitudes by b4,b8.
No cancellation between Gaussian coefficients can evade these amplitude caps.

## Fidelity uses normalization directly

For any nonnegative lambda4,lambda8, let w=lambda4 e4+lambda8 e8. After
phase-aligning phi to psi, Cauchy–Schwarz gives

    |<psi,phi>| <= ||psi-w|| + lambda4 b4 + lambda8 b8.

For the FK physical target the three hopping coordinates q are positive and
rigorously enclosed, so the norm on the right is
sqrt(q0²+(q4-lambda4)²+(q8-lambda8)²). Optimize the two multipliers numerically,
freeze them as exact binary64 values, and evaluate the entire bound with
directed arithmetic. The optimization need not be certified for the chosen
test vector to produce a valid global upper bound. Square and cap at one.

Equivalently, the optimal such bound is the support of the unit ball clipped
by the two amplitude caps. This is why it can be much sharper than converting
a single witness discrepancy into state distance and then into fidelity.

## Energy uses spectral penalties

In the hopping singlet basis the Hamiltonian is exactly

    [[-14-4t, -3sqrt14, 0],
     [-3sqrt14, -50, -3sqrt14],
     [0, -3sqrt14, -14+4t]].

For mu4,mu8>=0, adding mu4|e4><e4|+mu8|e8><e8| gives

    <phi,H phi> >= lambda_min(H+penalties)-mu4 b4²-mu8 b8².

The independently certified full-spectrum first excited lower bound exceeds
E_G at all three tested t. The penalized singlet minimum is <=E_G by the
vacuum trial vector. Thus its 3x3 minimum is the FULL Fock-space minimum,
including the odd sector and all nonsinglets. This reduction is not an
assumption that the approximate state itself is a singlet.

Choose nonnegative penalties by a small numerical search, then bracket the
minimum using exact rational Sylvester minors: both off-diagonal squares are
126. The three leading minors are positive at the recorded rational lower
endpoint. Subtract the penalty caps and the physical ground-energy upper
endpoint with directed arithmetic. This produces a rigorous energy-error
lower bound independently of any Gaussian variational optimization.

## Actual finite-dictionary comparisons

Each window dictionary contains 97 actual Gaussians: the hopping vacuum,
four eight-vertex mass/hopping orbits, and two radii on 32 generic quadratic
unitary directions. All atom energies and Gaussian purities are checked.

Fidelity maximizes a linear target overlap under norm<=1 and coefficient
l1<=B. The norm-ball solution is subsequently normalized and its actual cost
checked. When this solution lies on the unit sphere, the convex relaxation
also solves the finite-dictionary fidelity problem, up to solver accuracy.
An inaccurate solver status is preserved, not advertised as a certificate.

Energy uses four initial states and monotone feasible inverse iteration. Each
subproblem minimizes <phi,(H-E0+1)phi> subject to
Re<phi_previous,phi>>=1 and coefficient l1<=B. Its state has norm>=1, so
renormalizing preserves the cost budget and cannot increase shifted energy.
This gives feasible numerical variational states; it does not prove a global
finite-dictionary energy optimum. Saved coefficients are independently checked
in full Fock space against all constraints and all global lower/upper bounds.
