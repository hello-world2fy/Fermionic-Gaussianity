# Experiment A — Bosonic Hartree / Coherent-Frame Extension Validation

Independent end-to-end check of lifting the fermionic Gaussian convex-synthesis frame
(`Gaussian_Convex_Synthesis_Algorithm.pdf`, `paper_v2/sections/dual_core.tex`) to bosons:

* `H = Sym^N C^m`, dimension `D = binom(N+m-1, m-1)`
* Hartree (product-coherent) rays `|g^N> = (sum_j g_j a_j^dag)^N/sqrt(N!)|0>`, `g` on the unit sphere of `C^m`
* frame `q_k(u) = d_{N,k} int |<g^N|u>|^{2k} dmu`, with `d_{N,k} = binom(Nk+m-1, m-1)`

## What was actually run (not just formula curves)

| File | What it does |
|---|---|
| `code/boson_frame.py` | Exact Dirichlet/Gamma moments + gauge-fixed deterministic quadrature (FFT-exact phases, Gauss-Jacobi simplex). No sampling. |
| `code/exp1_moments.py` | `q_k = multinomial(N,n)^k/multinomial(kN,kn)` vs independent Gamma integral, m=2,3,4; B_k^2->xi convergence. |
| `code/exp2_convex.py` | Real convex program `min J_k(u)=q_k(u)/2k-Re<u|psi>` (L-BFGS-B) rediscovers the analytic witness; cvxpy SOCP discrete-dictionary upper bound on `tau`. |
| `code/exp3_bosehubbard.py` | Exact diagonalization of unit-filling Bose-Hubbard (M=2,3), then convex synthesis, condensate fraction, one-site entanglement, certified `xi` interval. |
| `code/run_tests.py` | Re-runnable sanity suite -> `tests/test_log.json`. |

## Headline results

* q_k closed form vs exact integral: worst relerr **5.1e-14** over 864 (state,k) pairs.
* Convex optimizer KKT residual **<= ~1e-10**, direction error <= 1e-10 (rediscovered `u_k = q_k(n)^{-1/(2k-1)}|n>` from a random start).
* Certified sandwich `t_k^2 <= xi <= B_k^2` holds for every highlighted multi-occupancy state, k=1..40; gap ~8-21% at k=40.
* Discrete-dictionary SOCP upper bound on `tau=sqrt(xi)` tightens to the explicit lower bound: m=2 to ~1e-7; m=3 bracketed but still coarse.
* Bose-Hubbard: weak U/J ground state is a near-condensate (F_1~0.99 estimate), strong U/J depletes it (F_1~0.58) and lowers entanglement; the certified `xi` interval (test-vector lower, Hoelder/residual upper) widens with interaction.

## 1-RDM and condensate fraction (exp4)

`exp4_rdm.py` builds the true one-body RDM `gamma_ij = <a_i^dag a_j>` in the **full N-boson Fock space**
(not N_up), from small exact-diagonalization ground states saved to `results/bh_states.npz`.

* `trace(gamma)=N` (err ~4e-16), Hermitian (err 0), PSD (min eig ~-2e-16, numerical noise only).
* **Condensate fraction** (Penrose-Onsager) = `lambda_max(gamma)/N`.
* **Endpoints**: U=0 free hopping -> pure Hartree condensate, fraction = 1.0 exactly;
  analytic unit-filling Mott `|1,1,...,1>` -> `gamma=I`, fraction = 1/M (M=2:0.5, M=3:0.333).
* **F1 vs condensate fraction are DIFFERENT metrics**: at M=2 they coincide to ~1e-4; at M=3, U/J=6
  condensate fraction = 0.670 but F1 (max Hartree fidelity, local estimate) = 0.581. We report both
  side-by-side without claiming a scaling relation or a size/critical transition (N=M=2,3 only).

## Certified bounds and the uncertified part

Revised after independent review (`../review/a_findings.md`):

* **Strict lower bound** (KKT-free, any feasible witness `u`):
  `xi >= (|<u|psi>| / q(u)^{1/(2k)})^2`. Uses only the polynomial-exact `q(u)`.
* **Certified upper bound**: `tau(psi) <= d^{1/(2k)} q(u)^{(2k-1)/(2k)} + sum_n |r_n|/sqrt(F1(n))`,
  `r = psi - C(u)`. The ensemble-mass term is the Hoelder envelope (no odd-power integral);
  the residual term uses the explicit Fock atomic norms `tau(|n>)=1/sqrt(F1(n))`.
  `q(u)` and `C(u)` are trig-polynomial integrals -> exact to floating point (not interval arithmetic).
* **Numerical, NOT certified**: `B_k` from the odd-power quadrature `|<g|u>|^{2k-1}` is *not* a trig
  polynomial (m=3,k=1 rel err 3.8e-4 vs the analytic Fock value). Reported only as an estimate
  with a Richardson refinement margin.
* **F_1(psi) for interacting states** is a multi-start local sup -> numerical estimate / lower bound.
* **Fock-state analytic sandwich** `t_k^2 <= xi <= B_k^2` still uses the exact `Bk_closed` Gamma
  formula (no quadrature) and remains trustworthy.
* **Uncertified**: true `xi` lies in the certified interval; the gap widens with the KKT residual
  and interaction strength and closes only with smaller residual / larger `k`.

## Reproduce

```
VENV=../.venv
$VENV/bin/python code/exp1_moments.py
$VENV/bin/python code/exp2_convex.py
$VENV/bin/python code/exp3_bosehubbard.py
$VENV/bin/python code/make_figures.py
$VENV/bin/python code/aggregate_summary.py
$VENV/bin/python code/run_tests.py
```

Single process, single thread (OMP/OpenBLAS/MKL threads pinned to 1), seed `20261026`.
Old `gaussian_synthesis/` tree was read-only; no caches written there.
