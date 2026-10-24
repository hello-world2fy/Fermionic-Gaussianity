# A known-optimum two-contact test independent of FK

Take eight complex modes and

    psi=(3|vac>+|full>)/sqrt(10), w=|vac>+|full>.

A number-preserving change of basis puts any Gaussian with nonzero vacuum
overlap in four-pair BCS form. Let the four pair probabilities be p_j. The
absolute vacuum and full amplitudes are prod sqrt(1-p_j) and prod sqrt(p_j).
Group the pairs into two pairs of pairs and apply Cauchy–Schwarz. Each
squared two-component norm is

    (1-p_i)(1-p_j)+p_i p_j <= 1.

Equality in both this bound and Cauchy requires all p_j=0 or all p_j=1.
The zero-vacuum chart boundary follows by continuity (or trivially from the
full amplitude <=1). Therefore

    h_G(alpha|vac>+beta|full>)=max(|alpha|,|beta|).

For |alpha|=|beta|>0, the only contact rays are vacuum and full. They are
two isolated components, s_1=s_2=0. The actual two-Gaussian expansion
matches w's lower bound, proving tau(psi)=4/sqrt(10). It is not inferred
from sampling. Near either contact, -log|<w,g>|² has Fubini–Study Hessian
2 I_56, because the remote amplitude starts at fourth order in Thouless
pairing coordinates. The two local Laplace weights are equal by particle-hole
symmetry. This is a deliberately simple test of two-component competition.

## Exact integer moments from four radial variables

In the skew-symmetric Thouless chart Z, invariant Gaussian measure is
proportional to det(I+Z†Z)^(-7) times Lebesgue measure on 28 complex entries.
Skew Takagi reduction has four squared singular values x_j and radial
Jacobian prod_(i<j)(x_i-x_j)^4 prod_j dx_j. Each singular value occurs twice
in the determinant. Set p_j=x_j/(1+x_j). The factors (1-p_j)^(-14) from
the Jacobian cancel the chart density factors, leaving

    const * prod_(i<j)(p_i-p_j)^4 prod_j dp_j, 0<p_j<1.

The relative vacuum/full phase is uniform under the number-preserving U(8)
action. Expanding an integer power and integrating that phase first gives

    Z_k(alpha,beta) = sum_(l=0)^k binom(k,l)^2
       |alpha|^(2(k-l)) |beta|^(2l) M_(k,l),

where the Selberg integral ratio is

    M_(k,l) = product_(j=0)^3
      (l+2j)! (k-l+2j)! (7+2j)! / ((k+7+2j)! ((2j)!)²).

The factor d_(8,k) converts Z_k to q_k. All coefficients are exact rational
numbers in `fk5/two_contact_toy.py`. The chart density exponent can also be
checked from the fractional-linear action on complex antisymmetric Z; the
Jacobian exponent is 2(n-1). The missing chart boundary has Haar measure zero.

Two independent normalization checks are q_k(|vac>)=q_k(|full>)=1 for every
integer k (the exact Weyl dimension product) and q_1(w)=||w||². GPU validation
uses only the latter norm anchor, never the Selberg coefficients at k>=2.
The balanced moments tend to q_k(w)->2, one contribution from each isolated
contact; finite-k cross terms are included exactly by the polynomial.

## Canonical direction and the drift test

Write q_k(1,r)=sum_l a_l r^(2l). Define

    A(r)=sum_l ((k-l)/k) a_l r^(2l),
    B(r)=sum_l (l/k) a_l r^(2l-1).

The canonical reconstruction equations reduce to B/A=1/3 and
scale^(2k-1) A=3/sqrt(10). The script solves these at 90 digits and records
the equation residual and bisection bracket. The moment polynomial is exact;
the floating root bracket is NOT advertised as a directed interval proof.

This restriction is justified by symmetry: the target is SU(8)-invariant,
and the only SU(8) singlets in even Fock space are vacuum and full. Strict
convexity of q_k/(2k)-Re<psi,w> makes its minimizer unique, hence invariant.
Uniform relative phase makes q_k depend only on the two absolute amplitudes,
so the positive target fixes their positive phases. Strict convexity and
homogeneity also make B/A strictly increasing along r>0: its derivative has
the sign of the two-dimensional Hessian determinant. The scalar solution
therefore determines the full-space canonical witness, rather than a mere
restricted variational candidate.

The two contact dimensions and local weights agree, so

    (2k-1) log(r_k) -> log(1/3).

There is an O(1/k) height adjustment and NO dimension-induced log(k)/k term.
Thus the possible log(k)/k mechanism must not be asserted merely because
there are two contacts. This known-optimum test provides both exact moments
and a prescribed unequal component ratio without looking at an FK target.
