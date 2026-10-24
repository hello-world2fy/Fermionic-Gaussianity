# Conservative high-order bounds from exact lower-order projectors

Let Pi_k be the coherent-state projector in the full k-fold tensor product,
so q_k(v)=||Pi_k v^tensor(k)||². Its range is spanned by g^tensor(k) over
the complete even Gaussian orbit. For positive integers a,b,

`range(Pi_(a+b)) is contained in range(Pi_a tensor Pi_b)`.

Indeed every generating vector is g^tensor(a) tensor g^tensor(b), and each
factor belongs to the corresponding range. Both operators are orthogonal
projectors, hence Pi_(a+b) <= Pi_a tensor Pi_b as positive operators.
Evaluating on v^tensor(a+b) gives the exact inequality

`q_(a+b)(v) <= q_a(v) q_b(v)`.

Inductively, for k=m j,

`q_k(v) <= q_j(v)^m`, and `N_k(v) <= N_j(v)`.

Thus the independently verified integer q4 polynomial supplies a reliable,
usually loose, upper bound on every N_8,N_16,N_32,N_64,N_128. A direction
normalized with a directed q4 upper bound has q_k<=1 at all these orders.
This is also a globally feasible Gaussian dual witness. No high-order
integration estimate is used in that conclusion.

The certified target overlap is a lower bound on the direction's true
canonical ratio <v|psi>/N_k(v), as well as on tau_G. It is generally much
smaller than the Monte Carlo canonical L point estimate. Reports must keep
these quantities separate: a certified but conservative block bound does
not certify the point estimate or establish a small duality gap. The
direction may come from an imperfect optimization; its feasibility does
not rely on a small reconstruction residual.

The inequality does not by itself show monotonicity at arbitrary consecutive
orders or monotonicity of canonical B. Along integer multiples it does order
the fixed-direction N norms. Exact k=2,3,4 regression tests check the factors
and normalization of this projector argument independently of high-k sampling.

For exact canonical optimizers, L_k is the dual norm N_k*(psi). Pointwise
N_(mj)<=N_j therefore also proves L_(mj)>=L_j. This statement concerns
exact optimization and integer multiples; noisy point estimates need not
appear ordered, and it says nothing analogous about canonical B. These
facts are direct consequences of projector inclusion, not a novelty claim.
