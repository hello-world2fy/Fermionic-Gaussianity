# Contacts of the joint witness

This analysis concerns W_lambda = omega_S + lambda w0, lambda>4. It does
not certify the different two-orbit matching witness in A2.

Put a=2/(lambda-2) and b=sqrt(1-a²). The mass/hopping Gaussian g(a,n)
from TWO_ORBIT_CONSTRUCTION.md, n in S6, has covariance

    Gamma(a,n) = [[ b J(n), -a I8 ], [ a I8, -b J(n) ]].

The Clifford relations give Gamma²=-I. Spin(7) acts transitively on n,
fixes all three singlets, and preserves the phases used in that construction.
Every atom therefore has singlet projection

    A(a)=((2-a²)/4, -a/sqrt2, (2+5a²)/(4sqrt7)).

Consequently <omega_S,g>=(1+a)², <w0,g>=1-a²/2. Both inequalities
and their relative phase saturate simultaneously. Thus the full-dictionary
support is EXACTLY H_lambda=1+lambda+2/(lambda-2), not just bounded above
by this number. The eight simplex atoms already suffice to attain it.

## Exhausting the equality conditions

The following argument is for Gaussian rays; a common phase is free.
It uses the standard four-mode magic-basis and Gaussian Schmidt facts in
[Reardon-Smith, Supplement I–II](https://arxiv.org/html/2407.20934v1).
Their odd-local-parity version is obtained by a fixed local Majorana.

Equality in the strictly concave scalar envelope first forces sqrt(eta)=a.
Equality in both Schmidt inequalities forces all four sin(2 theta_j)=a.
Since 0<a<1, the Schmidt matrix is invertible. Equality in the nuclear-norm
polar pairing then fixes its polar unitary to the exact omega_S matrix W:
after phase alignment, M=W X with X positive. X is the square root of a
Gaussian reduced density matrix. Its four equal single-particle singular
parameters imply, up to its normalization, X=exp(r Q_J) for an orthogonal
complex structure J. Its occupation amplitudes have c>s>0 and 2cs=a.

The exact matrix identity W† reshape(w0)=2|M_local><M_local| turns magic
saturation into an expectation of X in the local magic state. In the
complementary-pair proof of PRODUCT_MAGIC_NECESSITY.md, the extreme pair
has weight c⁴+s⁴, while each middle even pair has weight 2c²s². Their
difference is (c²-s²)²=1-a²>0. Thus equality forces the local magic state
to have support ONLY on the vacuum/full pair for J, with equal magnitudes.
The opposite local-parity sector cannot saturate this extreme-pair bound.

In a real magic basis the local Gaussian vectors are (r+i s)/sqrt2 with
real orthonormal r,s. Fixing the phase of their maximal overlap with the
real unit magic vector M gives r=M, s=N with N a real unit vector in M's
seven-dimensional orthogonal complement. Hence these local contact rays
are exactly S6. Their vacuum/full pairs give the complex structures J(n)
spanned by the seven Clifford matrices. The already fixed polar unitary
determines the other cluster, so there is one S6, not two independently
chosen local directions. This yields precisely Gamma(a,n) above.

The coordinates n and -n give different global rays for 0<a<1. At a=1
the orbit collapses to the hopping Gaussian. The lambda=infinity endpoint
is a different witness (w0) with larger S6 x S6 contact set; it cannot be
classified by substituting infinity into the finite-lambda equality proof.

## Contact-only representation has a missing singlet direction

Every contact projects to the SAME A(a), up to its freely chosen ray phase.
For any superposition of these contacts, its singlet projection is in
span(A(a)). Thus an FK singlet z not collinear with A(a) cannot be exactly
represented using these contacts, regardless of the number of terms/cost.

For a normalized arbitrary contact-only superposition phi, Cauchy gives

    F(z,phi) <= |<A(a),z>|² / ||A(a)||².

This ceiling is attained by the normalized eight-point average A(a)/||A(a)||;
its actual coefficient cost is 1/||A(a)||. The interval computation in
contact_comparison_01 encloses this ceiling for the exact physical t20 state.
It is close to one but strictly smaller: high fidelity is not exact synthesis.

The second orbit of the certified 16-term expansion has eta=1, hence is
Spin(7)-inequivalent to the main orbit and the joint contacts. This does not
prove that precisely this second orbit must occur in every optimal expansion.
What is proved is the failure of the complete joint contact-only dictionary.

## Scalar concentration and the remaining distance question

For all exact decompositions with cost <=U, the existing joint envelope gives
a global bound on E[(sqrt(eta)-a)²]. The new smaller certified U can be used
without changing that proof. Actual finite coefficients test its slack.
Covariance distances to each known orbit can also be minimized explicitly
over n by aligning the relevant seven-dimensional Clifford projection.

A positive local normal Hessian supports a local quadratic stability law.
High precision by itself neither supplies a rigorous neighborhood radius nor
a global constant c for quotient distance. That stronger quantitative part
of B2 remains open; it is not inferred from the scalar concentration bound.
