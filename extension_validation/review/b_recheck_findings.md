# Experiment B (Slater) — revised noci re-check (independent)

Reviewer: independent, read-only. Evidence: `b_recheck.py` (this review dir).
Producer revised: `slater/noci.py` (convex-J), `summary.json`, H2 demo.

## VERDICT
The corrected convex-J solver is GENUINE. The ~0.95 residual on the Hubbard
local 40-atom dictionary is a SPAN limitation (r outside col(A)), proven
mathematically, NOT a solver defect — confirmed to machine precision. This
matches the producer's corrected narrative. AWAITING the Fock-complete +
non-orthogonal-atom supplement before any final pass on full-residual synthesis.

## INDEPENDENT VERIFICATION (b_recheck.py)
- **T1 orthonormal complete dict**: kkt_err=3.7e-7, proj_residual=8e-16, span=8.
  Solver reconstructs r; kkt_err is L-BFGS stationarity tolerance.
- **T2 non-orthogonal FAT dict (12-dim, 40 atoms, spans)**: kkt_err=8.1e-7,
  proj_residual=1.1e-15. **Proves the non-orthogonal convex-J solver reconstructs
  r exactly once the dictionary spans** — the method works, as the reviewer required.
- **T3 narrow dict (400-dim, 40 atoms)**: kkt_err=0.9527 == proj_residual=0.9527
  to |diff|=2.9e-14; residual orthogonal to col(A): ||A^T resid||/(||A||||resid||)
  =1.8e-16. So the 0.95 is exactly the un-spanned component, by construction.

## H2 demo (slater/results/h2_demo.csv)
- span_dim=4 (full 4-vector active space), proj_fci ~1e-15 at all three R,
  kkt_fci 2.5e-8/7.9e-8/4.1e-8. Gauge tau 1.898/1.834/1.789.
- E_FCI R=0.74 STO-3G = -1.85239 (matches known H2/STO-3G FCI). Self-consistent.
- This IS the spanning-dictionary case: convex-J synthesis reconstructs FCI to
  machine proj with kkt ~1e-8.

## STATUS / WHAT REMAINS (per reviewer instruction)
- The producer correctly retracted v1 power-reweighting (F1-F5), the "secant=convex
  sum" wording (F7/F8), and now honestly reports kkt_err==proj_residual on the
  local neighborhood.
- DO NOT reject the method on the 0.95 Hubbard residual: it is a finite local-dict
  span gap, and T1-T2 prove the solver itself is accurate when span(A) contains r.
- AWAIT: Fock-complete dictionary + non-orthogonal atoms on >=1 representative
  Hubbard case (L=6 U=8) across multiple seeds, demonstrating kkt_err -> small
  (proj_residual -> small) once span(A) covers the residual. Only then can B's
  full-residual synthesis claim be assessed. Until then: dimer extent = PASS
  (analytic, 1e-15); convex-J solver correctness = PASS (T1-T3); Hubbard
  advantage = UNRESOLVED pending the spanning-dictionary supplement.

## H2 energy labeling + finite gauge note
- E_FCI reported (-1.85239 @0.74, -1.35093 @1.5, -1.21323 @2.0) is the
  ELECTRONIC energy. Total = electronic + nuclear repulsion:
  @0.74 = -1.85239+0.71510 = -1.13728 Hartree (matches H2/STO-3G total ~-1.137).
- Finite atomic l1 gauge tau_fci=1.898 (0.74) > sqrt(2)=1.414: this is the
  finite-dictionary atomic mass, an UPPER bound on the continuous atomic norm,
  NOT an exact tau. No exact continuous-extent claim for H2; the 4-atom demo
  reconstructs FCI to proj~1e-15 but the reported gauge is a finite upper bound.

## RE-CHECK of Fock-complete complete_ci.py — INTERIM (awaiting 3 small fixes)
Pool A = all external orthonormal Fock columns (span(r) guaranteed) + 40 non-orth
Slater atoms around dominant Ritz det; 3 methods SAME pool, 1 atom/step, budget 8/12.
- proj_residual median 1.6e-15 (max 3.7e-15) -> Fock-complete span confirmed; so
  replica kkt_err median 1.4e-6 / max 3.3e-4 IS genuine optimization accuracy,
  not a span gap. Matches reviewer expectation.
- Same-pool / one-atom-order / Ritz-energy-vs-exact scoring is FAIR.
- FINDING: replica picked non-orth atoms 0/132 (greedy_noci 90/132). The max|c_a|
  convex-J ranking NEVER selects a non-orthogonal Slater atom -> de facto replica
  = Fock-column greedy; the non-orth mechanism is not exercised by this ranking.
- Energy dE@budget12 (mean over 4 cases x 3 seeds): greedy_noci 1.38 < replica
  1.94 < cipsi 3.11. No energy advantage of replica over greedy at matched budget.
- AWAITING producer fixes (explicit final scope): (1) E n_det off-by-one accounting
  (E is pre-append Ritz; row n_det=2 E=0 is really n=1); (2) energy-weighted
  coefficient ranking (abs|c| is not energy-aware); (3) norm-projection duplicate
  rejection instead of min-overlap (remove QR dependence). Re-verify after.

## B FINAL ledger (576 rows, n_det 1..12, 4 cases x 3 seeds)
Methods: cipsi, greedy_noci, replica_coef (ablation), replica_energy.
- Fixed-budget mean dE vs exact:
  n=8:  cipsi 2.35, greedy 1.57, replica_coef 2.29, replica_energy 3.52
  n=12: cipsi 1.79, greedy 1.32, replica_coef 1.86, replica_energy 1.80
- Paired lowest-dE wins over 24 (tag,seed,budget8/12) groups:
  cipsi 10, greedy 7, replica_coef 5, replica_energy 2.
  => replica (either ranking) does NOT beat CI/greedy at matched budget.
- KKT: replica_coef median 1.4e-6/max 3.3e-4; replica_energy median 1.7e-6/
  max 2.3e-4; proj_residual max 5e-15 (Fock-complete span). Genuinely converged.
- picked_nonorth = 0/144 for BOTH replica variants: max|c| and energy-aware
  ranking still never selects a non-orthogonal Slater atom. The non-orthogonal
  Slater mechanism is never exercised by replica; it is de-facto Fock-column CI.
- Time is cumulative within step; excludes next-step overhead boundary (noted).
- VERDICT: convex-J solver correct & converged, but no energy/walltime advantage
  over PT2/greedy; non-orth mechanism inactive. Dimer extent analytic = PASS.
