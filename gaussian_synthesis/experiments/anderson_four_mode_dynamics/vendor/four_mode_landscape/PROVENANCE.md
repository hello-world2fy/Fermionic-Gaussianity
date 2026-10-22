# Newton solver provenance

`exact_solver.py` is a byte-for-byte snapshot of
`gaussian_synthesis/experiments/four_mode_landscape/exact_solver.py`
from the existing local commit `e5aa60b` (Add isolated four-mode Gaussian landscape experiments and results).
That separate experiment commit was not on remote `main` when this branch was prepared.
Only this necessary solver is included here; the unrelated landscape experiment is not a dependency.

SHA-256: `03ec40f3e8eaeda0edbda02ce28cda5d88da38c335142d23a79cbf4d01a1fc71`

The numerical solver has not been modified. `_bootstrap.py` is a publication-only import adapter
to this experiment's repository-relative setup. The original setup file and experiment scripts
are preserved in `../../results/source_snapshot/` for comparison with the historical result hashes.
The general `gaussian_synthesis` API, Haar sampler and coefficient measure still come directly
from the repository's `src/gaussian_synthesis/` package.
