# Canonical radial elimination

Let v be a nonzero direction, phase-aligned so lambda=<v|psi> is positive
and real. Define the exact (possibly implicit) radius

`r(v) = (lambda/q_k(v))^(1/(2k-1))`, `u_eff = r(v) v`.

Homogeneity and the normalized tilted Gaussian measure give

`C_k(v) = q_k(v) E_(pi_k,v) [g/<v|g>]`.

Consequently

`C_k(u_eff) = lambda E_(pi_k,v) [g/<v|g>]`,

`B(u_eff) = lambda Z_(k-1/2)(v)/Z_k(v)
          = lambda / E_(P_k,v) |<g|v>|`,

`L(v) = lambda / q_k(v)^(1/(2k))`.

The first two expressions contain no estimated partition function. This is
an analytic change of parameterization, not a claim that a noisy saved
numeric radius has become exact. The stored v and the exact integral
definition specify u_eff; a numerical radius remains an uncertified estimate.
P is unchanged by positive radial scaling. The remaining reconstruction
residual tests the angular canonical equation. Full-Fock Spin(7) and time
reversal averaging are applied exactly as in the original estimator.

Eliminating the radius from the convex objective gives a positive monotone
transform of maximizing lambda/q_k(v)^(1/(2k)). Thus the direction found by
the original convex training problem can be reused; its empirical scalar
normalization is a nuisance for the direction equation. Residuals and five
independent seed checks are still necessary, and failures are retained.

`tests/test_radial.py` checks homogeneity and the resulting full tensor
reconstruction against the independent exact k=2 backend, and checks the
gradient relation with the verified exact k=4 Cartan matrix. It also checks
invariance under changing the numeric scale of the direction.

Reanalyses live in new directories. Their original observations, old gates,
old numeric-u residuals, and full P files are preserved and content hashed.
The P expectation is independently recomputed from every saved ket, rather
than trusting cancellation of two printed partition values. High-order
point estimates of N_k and L still lack directed error bounds. Separate
conservative N_k upper bounds follow from exact lower-order Cartan blocks
(CARTAN_BLOCK_BOUNDS.md). No stronger dual certification is inferred from
radial elimination or empirical reconstruction gates.
