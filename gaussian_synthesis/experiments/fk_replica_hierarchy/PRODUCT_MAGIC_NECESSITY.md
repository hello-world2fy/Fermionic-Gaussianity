# Sharp physical composition constraint at t=0

For every normalized eight-mode Gaussian g, partitioned into the two
four-mode FK clusters, the following sharp bound holds:

`|<2 M_A M_B|g>| <= 1 - eta(g)/2`,
`eta=||Gamma_AB||_F²/8`.

This is a global bound on the complete Gaussian dictionary. It is stronger
than knowing only the maximum overlap, and is not inferred from the sampled
contact set. Equality is achievable at every eta in [0,1].

## Derivation

The Gaussian Schmidt decomposition uses four mode pairs with coefficients
`t_y=product_j c_j^(1-y_j) s_j^y_j`, where c_j²+s_j²=1. Local Gaussian
unitaries can be chosen parity preserving. Each transformed maximally magic
four-mode state has conjugate-paired occupation amplitudes, hence
`|a_y|=|a_(complement y)|`, and the same is true for b on the second cluster.
Only even y contribute. These standard Schmidt and magic-basis facts are
summarized in the supplement to
[Reardon-Smith, arXiv:2407.20934](https://arxiv.org/html/2407.20934v1),
sections I–II. The following quantitative eta bound is derived here.

Group the eight even bitstrings into four complementary pairs. Put
`R_j=sqrt(2)|a_y|`, `S_j=sqrt(2)|b_y|`. Both R and S are unit real vectors.
The triangle inequality followed by Cauchy gives

`2 |<M_A M_B|g>| <= sum_j (|t_y|+|t_complement|) R_j S_j`
`                    <= max_even_y (|t_y|+|t_complement|)`.

For any selected pair, exchange c_i and s_i according to its bitstring, and
write p_i=s_i² and pbar=mean p_i. AM–GM gives

`product c_i + product s_i <= (1-pbar)²+pbar²`.

Meanwhile `eta=mean sin²(2 theta_i)=4 mean[p_i(1-p_i)]`, and
`mean[p_i(1-p_i)] <= pbar(1-pbar)`. Combining these inequalities yields
`product c_i+product s_i <= 1-eta/2`.

The saturation arc already constructed in ENERGY_WINDOW_THEORY.md has
`eta=(1-x)²` and `<s0|g(x)>=1/2-eta/4` for x in [0,1]. It therefore attains
the bound throughout the entire eta interval. The endpoint eta=0 recovers
the exact full-dictionary support h_G(2s0)=1 and atomic norm tau(s0)=2.

## Every low-cost decomposition

For an arbitrary exact phase-aligned coefficient measure Lambda with cost B,

`2 <= integral |<2s0|g>| dLambda <= B - (1/2) integral eta(g) dLambda`.

Consequently, if B<=U,

`E_(Lambda/B)[eta] <= 2(1-2/U)`,
`Lambda{eta>=eta0}/B <= 2(1-2/U)/eta0`.

These bounds are deterministic and apply to all decompositions, not only the
canonical measure. For example, U=2.02 and eta0=0.2 force the coefficient
mass fraction with eta>=0.2 below 9.91%.

At the exact optimum B=2, eta=0 almost everywhere. A pure Gaussian with zero
cross covariance factors across the two clusters. Equality of its target
overlap then forces both factors onto their four-mode magic contact S^6.
Thus every optimal decomposition is supported on the product contact manifold
S^6 x S^6 (real dimension 12), up to alignment phases. This agrees with the
12 null directions measured independently in the full 56-dimensional
Gaussian tangent Hessian. The null-direction count itself is a numerical
cross-check; the product-support conclusion follows from the inequality.

At t=0 symmetry forces each exact finite-order canonical witness to be a
positive multiple of s0. This analytic support envelope can therefore
normalize every finite-order witness to 2s0. It does not imply monotonic
canonical costs or extend the exact contact classification to t>0.

## Exact canonical reconstruction and finite realization at t=0

The two independent local Spin(7) stabilizers have a unique invariant line
in the global even sector, spanned by s0. (The local magic is odd in this
repository's Fock convention.) Equivariance gives C_k(s0)=q_k(s0)s0 for
every positive integer k. Thus u_k=q_k(s0)^(-1/(2k-1))s0, while its canonical
P_k depends only on the known direction s0. Its mass is

`B_k = Z_(k-1/2)(s0)/Z_k(s0) = 1 / E_(P_k) |<g|s0>|`.

There is also an explicit finite projector, independent of this continuous
symmetry argument. The local target is exactly (e7-e8)/sqrt(2). The 16
signed even Majorana monomials listed in `fk4/product_magic_twirl.py` sum
to 16|M><M|; the code checks this identity using Gaussian-integer matrices.
Their 256 tensor products sum to 256|s0><s0|. Every branch is an actual
even Gaussian unitary, up to ray phase.

This finite identity also proves the canonical direction statement without
requiring a classification of the continuous stabilizer: every branch fixes
s0, Haar equivariance forces it to fix C_k(s0), and averaging the branches
forces C_k(s0) into the one-dimensional range. Taking its s0 overlap gives
the scalar q_k(s0).

For any N Gaussian particles with nonzero a_i=<g_i|s0>, define
`b=1/mean_i |a_i|`. The actual 256N Gaussian branches U_ab g_i with
coefficients `b a_i/(256 N |a_i|)` reconstruct s0 identically and have
cost b. No averaged projected vector is counted as a Gaussian atom.
For canonical P samples, b estimates B_k and is independently the cost
of this finite realization. Finite-sample distribution error remains;
the estimated q_k and numeric canonical radius are not thereby certified.
Floating ket arithmetic and costs are reported separately from the exact
integer projector identity. The original noisy k8 runs remain unchanged.
