# Experiment B — Slater / NOCI convex-J residual synthesis

Numerical validation of the "replica convex synthesis" idea on Hubbard chains
and a tiny H2/STO-3G active space. All code lives here; old project files are
untouched.

## Environment
Use the shared venv (do NOT reinstall):
```
/Users/bytedance/Desktop/Fermionic Gaussianity/extension_validation/.venv/bin/python
```
Python 3.14; packages: numpy, scipy, matplotlib, pymupdf, pyscf (see
`molecule_requirements.txt`).

## Reproduce
```bash
PY=/Users/bytedance/Desktop/Fermionic\ Gaussianity/extension_validation/.venv/bin/python
$PY tests/test_slater.py        # 7 unit tests (dimer, convex-J closed form, H2 FCI)
$PY run_benchmark.py            # local non-orth dictionary benchmark (projected-local)
$PY run_complete.py             # Fock-complete + 40 non-orth Slater, budget 12, 3 seeds
$PY plot_results.py             # regenerate artifacts/*.png
```
Reviewer KKT check: `$PY ../review/b_noci_test.py`.

## Layout
- `hubbard.py`        half-filled open-chain Hubbard Fock-basis ED.
- `dimer_extent.py`   Takagi extent xi_S vs analytic 1+U/sqrt(U^2+16t^2).
- `noci.py`           non-orth Slater dictionary + true convex-J solver
                      (L-BFGS on row(A); analytic grad C(u)-r).
- `selected_ci.py`    local-dictionary loop (fock_greedy/pt2/noci_greedy/replica
                      + power_weighted_heuristic ablation).
- `complete_ci.py`    Fock-complete pool + 40 non-orth Slater, fair 3-method compare.
- `molecule.py`       H2/STO-3G active-space Hamiltonian (matches pyscf FCI <1e-9).
- `results/`          CSVs/NPZ; `results/diagnostics/` holds the earlier
                      projected-local benchmark for reference.

## Headline result (see summary.json)
- **Analytic dimer**: xi_S = analytic to ~1e-15.
- **Fock-complete feasibility**: convex-J KKT residual ~1e-6 (genuine, span=1),
  but at matched 8/12-determinant budget the convex-J atom selection collapses
  to CIPSI (identical energies) and costs ~5-10x walltime; no advantage.
- **Local projected dictionary**: spans only ~5% of the residual; also no advantage.
- **H2**: convex-J reconstructs FCI to 1e-8; reported gauge tau~1.8-1.9 is an
  UPPER bound on a restricted 20-atom dictionary, NOT the exact Takagi extent
  (which for N=2e/m=4 spin-orbs is <= sqrt(2)).
