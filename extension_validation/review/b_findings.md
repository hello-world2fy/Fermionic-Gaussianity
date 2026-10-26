# Experiment B (Slater) — independent review findings

Reviewer: independent, read-only. Evidence: `b_independent_check.py`, `b_noci_test.py` (this review dir).
Producer code: `extension_validation/slater/{selected_ci.py,noci.py,dimer_extent.py,hubbard.py}`.

## VERDICT
- dimer Slater extent analytic check: **PASS** (independently reproduced to ~1e-15).
- Benchmark core claim ("replica = genuine finite-k convex-J synthesis"): **FAIL as implemented** in `selected_ci.py`; the newer `noci.py` attempts it but its fixed-point iteration is mathematically broken (see F2/F3) and is not wired into the reported benchmark.

## PASS
- **P1 dimer xi_S = 1 + U/sqrt(U^2+16 t^2)**. Rebuilt the 2-site half-filled Hubbard dimer by hand (4 Sz=0 states), Takagi/nuclear-norm extent (lam1+lam2)^2 matches to max rel err 6.7e-16 over U in {0,...,64}; lam2=0 at U=0 (single Slater) -> 1/sqrt2 as U->inf. Energy E=(U-sqrt(U^2+16t^2))/2 also confirmed.
- **P2 baselines are strong & fair**: greedy (|residual|) and PT2 (Epstein-Nesbet |u|^2/(H_dd-E)) on identical pools, same k_new, wall-time split into candidate-generation / selection(+convex weights) / subspace eigh; selection overhead ~10-16 ms is included, not hidden.
- **P3 honest verdict**: producers themselves report no energy win over PT2 and do not advertise speedup.

## FAILURES / SPECIFIC REVISION POINTS
- **F1 `selected_ci.py` "replica" is NOT the convex-J stationary point.** `_score_candidates` never optimizes a witness w_k: no projected-SGD on J(w)=q_k/(2k)-Re<w,r>, no check ||C_k(w)-r||. It sets `p_d ∝ |u_d|^{2 k_rep}` directly on raw residual amplitudes `u_d=<d|r>` (selected_ci.py:84-96). That is a power-law reweighting, not the moment-map fixed point C_k(w)=r. The producer's own new docstring (noci.py:4-16) now concedes this.
- **F2 wrong KKT exponent in the old heuristic.** On an orthonormal pool the true stationary point C_k(w)=u gives w_d ∝ sign(u_d)|u_d|^{1/(2k-1)}, hence tilted atom weights pi_d ∝ |w_d|^{2k} ∝ |u_d|^{2k/(2k-1)}, NOT |u_d|^{2k}. Numerically (k=2): producer [0.636,0.016,0.005,0,0.082,0.260] vs correct [0.358,0.106,0.072,0.017,0.181,0.266]. (Only k=1 coincides.)
- **F3 `noci.py` convex_j_synthesize does NOT solve the convex program.** Iteration `w <- A(|z|^{2k-2} z)` then Euclidean-normalize (noci.py:120-128) is power iteration on grad F (F=Σ|<a,w>|^{2k}); its attractor maximizes F itself and is **independent of target r** (init w=r/||r|| is overwritten). Independent test (dim=12,m=6): achieved dual value <w,r>/N_k(w) = 0.215/0.058/0.005 at k=1/2/4 vs brute-force optimum 0.304/0.393/0.431; TRUE KKT residual ||r - t A c||/||r|| with t=<w,r> fixed = 0.79/0.99/1.00 (never converged). Correct method is projected gradient ascent on <w,r> inside the N_k ball (an ell^{2k} ball in coefficient space), or solve A(|z|^{2k-2}z)=mu r directly.
- **F4 reported kkt_err is misleading.** It least-squares-refits `alpha = <r,synth>/||synth||^2` and reports ||r-alpha synth|| (noci.py:137-140). At k=1 this shows 5e-3 while the true KKT residual (fixed t) is 0.79. The refit absorbs the non-convergence. The honest number is `proj_residual = ||r - A*lstsq(r)||`, which must be reported instead.
- **F5 rho diagnostic is broken (sign mismatch).** `rho=||uhat - p||` compares a SIGNED unit residual uhat to a NONNEGATIVE probability vector p (Σp=1, p>=0). Numerically rho grows 1.10->1.47 with k as pure sign-orthogonality; ||uhat||=1 vs ||p||=0.29-0.77. It is NOT the paper certificate rho=||psi - t s(w)||. Remove or redefine.
- **F6 dictionary was orthonormal Fock determinants in the benchmark.** The Schur frame d^S=prod(k+j-i)/(j-i), scalar N_k norm and continuous Grassmannian extent do NOT apply on an orthonormal pool (frame=identity); this is exactly why the weighting "collapses" to PT2 reweighting. `noci.py` does build genuinely non-orthogonal Slater vectors (rotated occupied subspaces, generalized 2x2 Ritz with overlap S=[[1,s],[s,1]] is correct) but this code is NOT run/benchmarked and NOT mentioned in summary.json (summary predates noci.py).
- **F7 narrative error: secant != convex mixture.** summary.json:17 "Y_R = R-th secant = convex/sum of R atoms" is wrong. A pure-state secant {Σ c_j g_j : c_j in C} is a COMPLEX LINEAR SUPERPOSITION (phases), not a convex hull {Σ p_j g_j, p_j>=0}. The convex object is the phase-balanced gauge L1 norm. A nonnegative p-vector Σ p_d e_d is a MIXTURE (density matrix), not a pure Slater superposition -- the producer's "reconstruction" Σ p_d e_d is not even a valid pure state. Rewrite all "convex mixture/secant=convex" wording accordingly.
- **F8 do not import unrelated secant theorems.** B's finite orthonormal selected-CI has no relation to continuous secant completeness (K_{3,7}=7, four-copy, W_{R,k} ideal generation). Conclusions must stay within the finite-pool operational claim; cite dual_core S6 only as motivation, not validation.

## NOTES
- noci.py is brand-new (01:35) and unreviewed by the producer; no result numbers use it yet. When they wire it in, require: (i) corrected convex solver, (ii) true fixed-t KKT residual, (iii) report proj_residual, (iv) run at matched budget vs PT2.
- No molecule benchmark (deferred, no pyscf) -- acceptable, not fabricated.

## B timed_bench — independent audit (252 rows)
- DEADLINE integrity: 0 rows with last_committed_time > deadline (commit checkpoint
  enforced). Cooperative, NOT hard-kill: a running L-BFGS/stage finishes; actual overrun
  max 0.108s, elapsed max 1.608s vs budget 1.5; rollback=True only 3 rows. FCI used for
  scoring only (dE=E-E_score); selection reads H/current Ritz/local pool, no psi_exact.
- CONCRETE H2 BUG: noci.rotated_slater(ham,Uup,Udn,th_up,th_dn) returns w === v EXACTLY
  on H2 for ANY angles (||w-v||=0, |<w,v>|=1), even anti-rotation. Verified at R=2.0.
  Consequences:
  (a) orbital_optimize 2x2 obj is flat in theta -> no improvement -> returns None;
      ALL 27 H2 rows stall completed_n=1, stage=orbopt_none (Hubbard works: n>1 35/36).
  (b) _starts seeds 1,2 "predetermined Thouless rotations" are also no-ops -> all 3 H2
      seeds identical (CSV confirms identical dE across seeds); NOT 3 independent starts.
  This is not "RHF Brillouin stationary" — the rotation itself does not move the Slater
  on the H2 (nu,nv)=(1,1) spin structure. orbopt_none 63/252 = 36 Hubbard-soft + 27 H2.
- orbopt_noci dE has 9 NaN rows (L4 systems) -> CSV needs allow_nan / finite guard.
- H2 same-n baseline not added yet (timed only).

## B final timed (456 rows) — independent replay
- Code-level LATENT checkpoint bug CONFIRMED by reading: committed tuple (line 134)
  stores (n,E,resid,time) but NOT psi; next-loop Ritz line 127 can overwrite psi, then
  SIGALRM fires; _save_ckpt (line 194, OUTSIDE try, after except) uses latest psi + E_last.
  Fix = save psi copy in the commit tuple and _save_ckpt uses that last committed psi.
- EMPIRICAL replay of all 17 on-disk ckpt npz: Rayleigh quotient <psi|H|psi> == stored E
  to ~1e-15 for ALL 17 (incl. all 9 surviving post_committed). => 0 of the saved ckpts are
  actually corrupted in THIS run; when SIGALRM fired it was during L-BFGS (psi stayed the
  committed Ritz psi), so no overwritten psi reached disk.
- Coverage: 456 rows claim a ckpt filename but only 17 npz on disk; 85 post_committed rows
  -> 9 ckpts on disk (76 missing, likely overwritten/cleaned) and cannot be replayed.
- Integrity: 0 rows with last_committed_time > deadline; 0 init_not_completed. SIGALRM
  active over whole timed region; stages budget_reached 154/full_space 158/post_committed 85/
  residual_zero 46/orbopt_none 13 (orbopt_none dropped 63->13 after complement-SVD/expm fix).
- H2 completed_n now {2,3,4} (was {1}); 3 rotated starts distinct. Fixed-time b=1.5s median
  dE all ~0 (many full_space); orbopt_noci mean 0.19 < greedy 0.25 < cipsi 0.55 < replica 0.81.
- Fixed-n 60-det fallback has no SIGALRM (cooperative); acceptable as same-n resource cap,
  label as such. No warmup recorded -> cold-start method bias; record explicitly.
