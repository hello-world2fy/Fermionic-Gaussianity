# Experiment C (signed Gaussian) — independent review findings

Reviewer: independent, read-only. Evidence: `c_independent_check.py` (this review dir),
imports `sggen/`. Producer: `extension_validation/signed_gaussian/`.

## VERDICT
The witness lower bound R(rho_M)>=3 and the white-noise lower line are correct.
But (i) the Hubbard R-curve is dominated by finite-dictionary bias (U=0 low-T
should be ~1 yet LP gives 1.86-2.04); (ii) the "ratio 7.42" witness is pool-span
unstable (denser pool -> 2.38, still shrinking); (iii) several text derivations
have sign/prefactor errors; (iv) R=3 is an upper bound ~3.000006, not an exact
analytic <=3 construction.

## FINDINGS / REVISION POINTS

- **C-F1 odd-sector value is 0, not -1.** Independently built W16=4| M16><M16|-P_+
  and evaluated on an odd Slater ket: Tr(W sigma_odd)=0.0 exactly (even rho_M and
  even projector both annihilate the odd state). The summary text "odd g give -1"
  / "odd block gives -1" is wrong. Does NOT change validity (sup is set by even
  g = 1), but fix the derivation.

- **C-F2 sup-overlap derivation drops the 1/2 prefactor.** |<M|g>|^2 = (1/2) N^2 |1+P|^2.
  The summary chain writes |<M|g>|^2 <= (1+mu1 mu2)^2/((1+mu1^2)(1+mu2^2)) <= 1/2,
  i.e. it bounds the prefactor-less fraction and then claims <=1/2; the fraction
  is <=1, and the 1/2 comes from the N^2/2 prefactor. Numeric sup over 2e5 Haar
  even rays = 0.4919 -> sup=1/2 confirmed, but rewrite the inequality with the 1/2.

- **C-F3 R(rho_M) is NOT certified exactly 3.** Witness lower = 3.0 exactly
  (Tr(W rho_M)=3, sup_even Tr(W sigma_g)=1). But the upper is a finite-dictionary
  LP: producers 3.000006; my finer contact dictionary (6000+400 atoms) gives
  3.000019, residual 1.4e-16. This is an UPPER bound with a ~2e-5 gap, not an exact
  decomposition. As the reviewer notes, an analytic finite-contact combination
  (real-orthogonal triality basis) should give TV exactly <=3; it is absent.
  State "R=3 within 2e-5 by matching bounds", not "R=3 verified".

- **C-F4 Hubbard R-curve finite-dictionary bias is ~2x at low T.** A free-fermion
  thermal even block (U=0) is a convex mixture of pure Gaussians -> true R~1 for
  all beta. Yet with an 8000-atom dictionary I reproduce LP R_norm = 1.0 (beta=1),
  1.77 (beta=4), 1.86 (beta=8), residual ~1e-15. So the U=0 curve (1.0->2.04 in the
  producers' dict) IS the discretization bias. The headline "interaction increases
  R" at fixed low T must subtract this baseline: e.g. at beta=8 the apparent
  excess over U=0 is 2.92-2.04=0.88, not 2.92, and even that is dictionary-dependent.
  Quantify: bias = LP_R(U=0,beta) - 1, ranges 0 -> ~1.0 over beta=0.5..8.

- **C-F5 Gibbs state is GRAND-canonical, and beta=0.5 is not max-mixed.**
  H includes -mu(n_a+n_b) with mu=U/2 and gibbs_state sums e^{-beta H} over all
  16 dims (N=0..4); the even block is then extracted. This is grand-canonical at
  the particle-hole point, NOT a canonical fixed-N ensemble. beta=0.5 != 0, so the
  state is not maximally mixed; the "hot_limit_R=1 / maximally mixed is Gaussian"
  framing is misleading. R_norm=1 at beta=0.5 means the LP recovered a dictionary
  atom exactly (residual ~1e-15), not that the state is the identity.

- **C-F6 witness ratio 7.42 is pool-span unstable.** W=-0.25I-1.25(D_a+D_b)+0.5h,
  Tr(W rho)=0.1067. Producers' 20k-pool sup_pool=0.0144 -> ratio 7.42. My denser
  1e5-ray pool already gives sup=0.0449 -> ratio 2.38 (product family gives 0), and
  ||W||=2.75 leaves the true continuous sup anywhere in [0.045, 2.75]. The ratio is
  shrinking toward 1 as the pool refines; it is NOT a stable certificate. The
  producers' own conclusion "no rigorous witness (finite-pool-only)" is correct,
  but flag that even the finite-pool number is not reproducible under pool
  refinement — target reconstruction (residual ~1e-15) is fine, the sup is not.

- **C-F7 white-noise plateau R=1 for p>=4/7 needs the continuum hull argument.**
  The witness lower line 3-7p/2 is exact. But claiming R=max(1,3-7p/2) exactly
  requires showing P_+/8 (and rho_p for p>=4/7) lies in the CONTINUOUS convex hull
  of Gaussians. The finite LP only upper-bounds; matching the line to 4 digits is
  consistency, not proof. State it as "consistent with" unless the continuum
  averaging argument for I_8/8 in the Gaussian hull is derived.

## PASS
- W=4rho_M-P_+ lower bound: Tr(W rho_M)=3, analytic sup_even=1 via (mu1-mu2)^2>=0,
  numeric sup 1.000000 -> R>=3 solid.
- LP machinery (signed TV over Hermitian-flattened atoms, HiGHS) is correct;
  residuals ~1e-15 confirm exact linear reconstruction on the dictionary.
- Honest limitations section already flags finite-dictionary and pool-sup caveats.

## RE-CHECK of C convex.py (operator J) — INTERIM (awaiting deterministic supplement)
- convex.py IS a real operator program, not LP: J_k(W)=d/(2k) E|Tr(W sigma_g)|^{2k}
  - Tr(W rho), grad = C_k(W)-rho, MC projected gradient. Gradient finite-diff
  max err 1.4e-11 / 1.8e-11 / 2.0e-11 (k=1,2,4) -> gradient CORRECT.
- C-A1 (user suspicion CONFIRMED): d_nk = spinor Weyl dim (8,35,294) has NO
  operator-Cartan basis for the reducible density-adjoint orbit; it is constant
  scaling. N_k=(d E|t|^{2k})^{1/(2k)} <= d^{1/(2k)} h is NOT a certified sup gauge.
  moment_witness (3.21/2.84/2.64) is optimistic; only sup_gauge_lower (2.32/2.43/2.57)
  is an honest estimate. Producer's own note acknowledges this.
- C-A2: MC KKT residual ||C_k(Wk)-rho|| = 0.044/0.053/0.108 (k=1,2,4) NOT converged.
  This is an optimization attempt / diagnostic, NOT an exact finite-dict verification.
  Retained as diagnostic only.
- USABLE now: R=3 bracket = analytic witness lower 3 + finite-LP upper 3.000006.
- product g(t,t): |<M|g>|^2=1/2 for ALL t (no t=1 special); earlier note withdrawn.
- Hubbard: added Slater eigvec atoms so U=0 even-block baseline R=1; odd block
  (N=1,3) not decomposed -> explicit limitation. No triality exact upper pursued.
- STATUS: feasibility PILOT, not a proof of all claims. AWAITING fixed finite-pool
  deterministic J + features J with converged KKT before final C verdict.

## C FINAL — CRITICAL flatten bug + valid bracket
- C-FLAT-CRITICAL: old magic_artifacts.npz weights (reported R_LP=3.000006,
  residual 1.8e-15) were solved with a 36-row (real-only) flatten. Independent
  recompute: FULL Frobenius ||recon-rhoM|| = 0.335, real part 2.5e-15, IMAG part
  0.335. The imaginary off-diagonals were never constrained -> recon != rhoM.
  The old upper 3.000006 is INVALID and must be discarded; the "residual 1.8e-15"
  was real-part only. Same suspect applies to the 01:48 hubbard curves (pre-fix).
- VALID now: corrected 64-dim flatten (run_exact_convex.py, M=550 pool, span rank
  64). rho_M finite-pool LP upper = 3.05398 with FULL Frobenius residual 8.6e-13.
  Analytic witness lower = 3.0. => R(rho_M) in [3.0, 3.054], bracket only.
- finite_pool_lower 1.82/2.29 = lower bound on FINITE-POOL extent, not continuum.
- hubbard feature J feat_error 63.9/13.3 (k=1/2) does NOT reconstruct rho; diagnostic.
  hubbard-star finite LP upper = 5.062 (single point, corrected).
- triality exact basis NOT built; non-Gaussian formal only -> cannot claim exact
  upper = 3. The [3,3.054] bracket is honest partial, not exact.
- Old MC convex_summary (KKT .04-.11) = failed convergence, diagnostic only.
- REQUIRED from producer before any stronger C claim: re-run magic + hubbard
  curves under the corrected 64-dim flatten; do NOT cite 3.000006 or old curves.

## C FEATURE off-diag factor-2 bug (reviewer, producer min-fix pending)
- run_exact_convex.py flattens raw Hermitian (diag once + offRe once + offIm once).
  Raw inner product feats.y = sum diag + sum offRe + sum offIm, but Tr(O rho) =
  sum diag Oii rhoii + 2*sum_off Re(Oij* rhoij). The off-diagonal factor 2 is
  MISSING -> hop_e covector undercounts by ~half; feature moments are not physical.
- Producer min-fix: covectors offdiag*=2, signed_mass=sum|atom c|, true feature
  error = ||Bf^T c - y|| (not unflatten(w)-rho). Short rerun ~0.13s.
- Consequence: full-rho finite coordinate LP synthesis (3.054, fresh 3.000028)
  REMAINS valid. The restricted feature witness is NOT physical scope -> cannot PASS
  until covector factor fixed and rerun.
