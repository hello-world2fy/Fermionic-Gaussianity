# Published experiment artifacts

This branch includes the experiment implementation, tests, original experiment specification, theory notes, execution state, configurations, manifests, verification records, reports, plots, CSV tables and JSON results. Start with [REPORT.md](REPORT.md) and [README.md](README.md).

The complete experiment directory occupies approximately 229 GiB locally. The following artifacts are intentionally excluded from Git and remain in the original local experiment directory:

- Raw NumPy particle, Gaussian-ket, kernel and checkpoint arrays (`results/**/*.npz` and `results/**/*.npy`).
- Per-run duplicate `source_snapshot` trees and Matplotlib caches.
- Two historical geometry summaries larger than GitHub's individual-file limit: `results/fixed_geometry_01/summary.json` and `results/fixed_geometry_01_qualified/summary.json`. The final geometry report and summary under `results/fixed_geometry_n8192_01/` are included.

Original local directory: `/home/tiger/Fermionic-Gaussianity-fk-two-cluster/gaussian_synthesis/experiments/fk_replica_hierarchy`.

The final four-order prediction summary, aggregate tables and cost audit are included. Historical failed-run records are preserved. Saved verification results describe the original full local artifacts; a fresh clone alone cannot rerun array-dependent verification without restoring or regenerating the omitted data. Manifests retain original paths and hashes; publishing does not rewrite scientific artifacts or their provenance.

No experiment or test was rerun for this publication. The completed run's 40-test log and independent verification records are included.
