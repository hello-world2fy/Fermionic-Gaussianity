# Experiment C — Signed Gaussian Operator Convex Synthesis

Independent re-implementation and verification for the four-mode fermionic
Gaussianity project. **Read-only** against `Gaussian_Convex_Synthesis_Algorithm.pdf`,
`paper_v2/`, and `gaussian_synthesis/src/`. Revised per independent review
`extension_validation/review/c_findings.md`.

## What is verified (honest brackets, not over-claims)

| Claim | Result |
|---|---|
| `rho_M = \|M><M\|` signed cost `R` | **Bracketed, not exact**: analytic continuous witness lower = 3; finite-dictionary LP upper = `3.000006` (~2e-5 gap) |
| `rho_p=(1-p)rho_M+p P_+/8` | LP upper **consistent with** `max(1, 3-7p/2)`, `pc=4/7`; plateau needs continuum hull argument |
| Witness `W=4rho_M-P_+` | Continuous certificate: `sup_even Tr(W sigma_g)=1`; **odd sector gives 0, not -1** |
| Convex program `J_k(W)` | **Real convex solve** (not LP): k=1,2,4; gradient finite-diff err ~2e-11; KKT residual reported |
| Hubbard U=0 sanity | R_norm=1.0000 at **all β** after adding quadratic-Gibbs Slater eigenstates |
| Few-observable witness | **No rigorous certificate.** Finite-pool ratio ~5.7 (was 7.4 at smaller pool) — pool-unstable; `||W||` bound gives <1 |

## Key honesty points (from review)

- **R=3 is bracketed, not proven exact.** Lower 3 is analytic; upper 3.000006 is a
  finite-dictionary LP. No finite analytic triality projector combination was found.
- **Overlap bound carries the 1/2**: `|<M|g>|^2 = (1/2)N^2|1+P|^2 ≤ (1/2)(1+μ1μ2)^2/
  [(1+μ1²)(1+μ2²)] ≤ 1/2` because `(μ1−μ2)²≥0`. Product BCS `g(t,t)` saturates for
  **all** t (μ1=μ2).
- **Odd sector**: both `rho_M` and `P_+` annihilate odd kets, so `Tr(W sigma_odd)=0`.
- **Hubbard dictionary bias**: without Slater atoms U=0 low-T LP gives ~2 (discretization).
  Adding U=0 quadratic eigenstates fixes the sanity; the interaction "excess"
  `R(U,β)−1` remains dictionary-dependent and is **not** a certified physical effect.
- **Grand-canonical**: Gibbs sums N=0..4 at μ=U/2; β=0.5 is **not** maximally mixed.
- **Finite-pool sup is not a certificate**: matching measured `f(g)=(1,⟨Oα⟩)` does not
  reconstruct the full state, and MC variance cost ≠ the sign problem. A `||W||` bound
  failing to certify does **not** prove no witness exists.

## Convex program (sggen/convex.py, sggen/synthesis.py)

```
J_k(W) = d_{n,k}/(2k) E_g |Tr(W sigma_g)|^{2k} - Tr(W rho),   k=1,2,4
grad   = d_{n,k} E_g |Tr(W sigma_g)|^{2k-2} Tr(W sigma_g) sigma_g - rho = C_k(W) - rho
```
Monte-Carlo projected gradient; Hermitian W flattened to real vector with the
correct Hilbert-Schmidt factor 2 on off-diagonals; gradient checked by finite
differences; KKT `C_k(W_k)=rho` residual reported. This is the PDF's convex
machinery for operators — **not** an LP.

## Layout / reproduce

```bash
VENV=/Users/bytedance/Desktop/Fermionic\ Gaussianity/extension_validation/.venv
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 $VENV/bin/python scripts/run_magic.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 $VENV/bin/python scripts/run_hubbard.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 $VENV/bin/python scripts/run_convex.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 $VENV/bin/python scripts/make_figs.py
PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python scripts/assemble_summary.py
```

Fixed seeds (20261026/27), single-threaded, ~90 s total. Float64, not interval-certified.
