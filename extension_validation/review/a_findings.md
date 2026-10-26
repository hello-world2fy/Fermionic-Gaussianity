# Experiment A (Hartree) — independent review findings

Reviewer: independent, read-only. Evidence: `a_independent_check.py` (this review dir).
Producer code: `extension_validation/hartree/code/{boson_frame,exp1_moments,exp2_convex,exp3_bosehubbard}.py`.

## VERDICT
Core frame/moment math and the Fock analytic witnesses are **correct**. The convex solve here is genuine L-BFGS on J (unlike B). Two certification gaps on the *interacting* Bose-Hubbard upper bound (quadrature error + missing residual widening); one lower-bound form should use the direct quotient.

## PASS
- **A-P1 864 moments**: reproduced exactly: (2,4):5+(2,6):7+(3,3):10+(3,4):15+(4,4):35 Fock states x k=1..12 = 60+84+120+180+420 = 864.
- **A-P2 q_k closed form**: q_k(n)=multinomial(n)^k/multinomial(kn) derived from Gamma-ratio sphere moments; matches independent integral to ~1e-15 (checked m=2,3,4). This is the correct Hartree Veronese frame d_{N,k}=binom(Nk+m-1,m-1).
- **A-P3 Fock exact witness**: u_k = q_k(n)^{-1/(2k-1)} |n> is the analytic KKT witness (diagonal-phase stabilizer fixes it to |n>); exp2 L-BFGS rediscovers it to ~1.5e-10 direction error. tau(|n|)=1/sqrt(F1(n)) exact.
- **A-P4 genuine convex solve**: exp2/exp3 minimize J=q/(2k)-Re<u,psi> with analytic gradient C-psi (moments by deterministic quadrature), KKT residual ~1e-9. This IS the finite-k convex-J stationary solve (contrast B's reweighting).
- **A-P5 lower bound is KKT-free**: t_k^2 = (|<u,psi>|/N_k(u))^2 is a valid xi lower bound for ANY feasible u; evaluated at the optimizer it tightens to t_k^2*.

## FINDINGS / REVISION POINTS
- **A-F1 eval_Bk quadrature is NOT exact for |a|^{2k-1}.** The docstring claims "all phase integrals over trigonometric polynomials -> trapezoidal exact". True for q_k=|phi|^{2k} (integer power, a trig polynomial), but FALSE for B_k=|phi|^{2k-1}=(phi phi*)^{k-1/2}, which is not a trig polynomial. On the exact Fock witness, comparing eval_Bk (quadrature) to Bk_closed (analytic Gamma) gives rel error: m=3,k=1 -> 3.8e-4; k=2 -> 1.0e-5; k=4 -> 8.5e-8 (m=2 stays ~1e-15). So exp3's interacting-state upper bound B_k^2 carries an unquantified ~1e-3..1e-4 relative quadrature error. Fix: widen B_k^2 by an explicit quadrature-error margin (or Richardson/analytic moment), and stop calling the odd-power integral "exact".
- **A-F2 exp3 upper bound omits the KKT-residual widening.** v1 certificates use t_upper = q^{exponent} + c*rho with rho=||psi-C(u)||. Here the tilted ensemble from ustar reconstructs C(ustar), not psi; the correct upper bound is [B_k(ustar) + c*rho]^2 (c=sqrt(D) d^{-1/(2k)}), not B_k(ustar)^2. Numerically rho~1e-9 so the gap is negligible now, but as a *certified* upper bound the residual term must be included/stated.
- **A-F3 lower-bound form should use the direct quotient.** They report t_k = q^{(2k-1)/(2k)}, which equals the witness quotient ONLY at stationarity C(u)=psi (Re<u,psi>=q). Rigorous KKT-free form is |<u,psi>|/q^{1/(2k)}. On a non-stationary u these diverge (checked: q-form 8.4 vs quotient 0.18 at k=2). At their converged ustar they coincide, so the number is fine, but report the direct quotient so the bound stays valid if optimization stops early or KKT residual grows.
- **A-F4 F1(psi) for BH states is a sampled/local sup, not a global certificate.** max_overlap_g uses 300 random starts + projected gradient; it can underestimate the true max_g. Report condensate_F1 as an estimate/lower bound, not an exact F1.
- **A-F5 no kkt residual column in summary rows**: exp3 rows report only t_k^2/B_k^2, not the per-witness KKT residual or quadrature margin; needed to trust the interacting-state interval.

## NOTES
- Fock-state sandwich t_k^2 <= xi <= B_k^2 uses analytic Bk_closed (exact) -> trustworthy.
- SOCP net tau upper bound direction is correct (net relaxes |<y|g>|<=1 over all g to a subset, so tau_SOCP >= tau_true) and tightens to sqrt(xi) with net density.
- Scaling limitation (D dense) is honestly stated.

## RE-CHECK of A revision (post a_findings) — ACCEPTED
Producer revised exp3 per findings. Read-only verification:
- Lower: xi_lo = (|<u,psi>|/q(u)^{1/(2k)})^2 — KKT-free quotient (F3 fixed).
- Upper: xi_hi = [d^{1/(2k)} q(u)^{(2k-1)/(2k)} + sum_n |r_n| tau_n]^2,
  r=psi-C(u), tau_n=1/sqrt(F1(n)) (F2 fixed; Hölder ensemble + Fock-atomic
  residual widening, matching v1 certificate structure).
- B_k odd-power quadrature demoted to numerical estimate with Richardson margin
  (B2-B1)/B2, no longer in the certificate (F1 fixed).
- F1(psi) labeled multi-start lower bound (F4); KKT_residual reported per witness (F5).
- Brackets consistent: M3 U/J=0.5 [1.048,2.801] KKT=1.5e-4 tau_res=6.7e-4 (explicit
  residual widened in); U=2 [1.417,3.785]; U=6 [2.143,5.725]. All lo<=hi.
- CAVEAT (floating point, not interval): algebra is rigorous but evaluated in
  double precision; KKT residual up to 1.5e-4 (M3U.5) means the bracket carries
  numerical slack not bounded by interval arithmetic. No claim of certified-width.

## A exp4 RDM — independent recheck PASS
Rebuilt gamma=<a_i^dag a_j> from saved bh_states.npz psi (independent loop):
- M3 UJ=6: trace=3.0 exactly, herm_err=0, minEig=0.341 (PSD), fraction=0.669949 (matches).
- M3 UJ=0: fraction=1.0 (pure Hartree endpoint), minEig ~0.
- Analytic Mott |1,...,1>: fraction m=2 -> 0.5=1/2, m=3 -> 0.333=1/3, trace=N exact.
- F1_hartree_estimate 0.5806 != condensate_fraction 0.6699 at M3U6 (diff 0.089) — correctly distinct.
- entropy uses log2 (bits); M3U6 S=0.5528 bits.
- LIMITATION: exp4 adds M2 UJ=0.5/2/6 states but the certified extent bracket [2.143,...]
  in summary.json is for M3 UJ=6 only; no certified extent for M2 states (RDM/fraction only).
- Local many-body matrix math kept separate from condition/PSD checks; no mixing.

## A exp4 RDM — RAW (unsymmetrized) check
Producer symmetrizes gamma=(G+Gdag)/2 before return (herm_err=0 by construction).
Independent RAW accumulation (no symmetrize) on M3UJ6: |G-Gdag|=8.3e-33 (machine zero),
trace=3.000000000000 exact, imag trace~1e-32. => raw accumulation IS Hermitian to machine
precision; the symmetrization is harmless, not masking a non-Hermitian bug. PSD minEig=0.341,
fraction=0.669949 unchanged.

## A exp4 FINAL strict JSON — verified
exp4_rdm.json parses STRICT (parse_constant rejects NaN/Infinity) -> no Infinity.
10 rows; Mott endpoint UoverJ=null (not Infinity). trace_err <=4.4e-16 across all;
fractions match independent rebuild (M3U6 0.6699, Mott 1/2=0.5, 1/3=0.333). Bound joins
same (M,N=m,UJ,J=1) with 5 finite points; M2U6 no extent bracket (RDM only);
U0 analytic xi=1, Mott analytic 2/4.5 endpoints. A exp4 = PASS.
