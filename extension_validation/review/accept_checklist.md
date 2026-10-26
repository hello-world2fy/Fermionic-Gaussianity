# Acceptance checklist — A/B supplementary design (read-only review of existing timing code)
Scope: upcoming fixed-deadline B benchmark + orbital-param greedy NOCI; A 1RDM/fraction extent.
Read-only; old project untouched. Results pending — no final review yet.

## B — fixed-deadline benchmark (traps found in existing code)
- LEAKAGE: run_benchmark.run_h2 line 113 builds residual r = H@guess - E*guess with E =
  ham.exact_ground() E_exact. That is target leakage. New design MUST build residual from the
  guess's own Ritz/Rayleigh energy, never E_exact/psi_exact.
- SCORING-ONLY access to psi_exact is OK (dE = E-E_exact, overlap |<psi,psi_exact>|). Verify
  candidate selection reads only H, current Ritz E, and local pool — never psi_exact.
- RESIDUAL in run_selected_ci line 110 = H@psi - E*psi uses current Ritz E (correct); keep that.
- DEADLINE: old driver is budget loop (max_det), has no deadline. New run must:
  * commit a checkpoint whose timestamp <= hard_deadline (wall clock, not CPU);
  * roll back / discard a step that crosses the deadline (not scored);
  * record actual_stop - hard_deadline overrun explicitly.
- TIMERS: existing t_pool/t_residual/t_optimize/t_ritz exist. Must ADD a separate orbital-opt
  timer for the new orbital-param greedy NOCI baseline. Clarify that H@psi residual matvec and
  H/eigh setup: generic shared H assembly + FCI scoring NOT counted; method prepare (pool,
  candidate gen, orbital opt, convex opt, Ritz on the small active space) IS counted.
- TIERS: 3 seeds / 3 starts / small-secs budget levels; report per-cell timing, not only final.
- TERMINATION: must stop and score when rnorm=0, dictionary spans r (full span), or no
  independent candidate remains. Old loop only breaks on cand.size==0.
- NEW BASELINE: orbital-param greedy NOCI must actually optimize orbital rotations
  (unitary parameters), not just the finite-pool 2x2 Ritz rank already in noci.two_dim_ritz_drop.

## A — 1RDM / fraction / F1 extent
- Build γ1 from the N-body BH state: verify trace(γ1)=N, γ1=γ1† (Hermitian), PSD (eigvals>=0).
- fraction = λmax(γ1)/N in [0,1]; report alongside.
- U=0 Mott half-filled sanity anchor: γ1 ~ occupation diagonal (localized), fraction ~ 1/N.
- F1 entropy-extent curve must reuse the SAME F1(psi)=sup|<g^N|psi>|^2 definition as the
  certificate; sampling grid consistent with exp1.
