# New hopping and interaction perturbations: frozen transfer test

All 2400 predictions across 160 Hamiltonians were saved and hashed before the first new full-spectrum ED. The Hamiltonians were then independently reassembled, every trial state checked for Gaussian atoms and normalization, and energy, fidelity, mass and Wick errors evaluated. No dictionary was modified using the validation targets.

Means across the preregistered couplings, amplitudes and five perturbation patterns (not independent error bars):

| Perturbation | R | Method | Mean fidelity | Mean energy error | Mean coefficient cost | Mean prediction seconds |
|---|---:|---|---:|---:|---:|---:|
| hopping | 16 | two_orbit | 0.99958693 | 0.021390134 | 1.7147801 | 0.00047102438 |
| hopping | 16 | physics_prior | 0.95518894 | 1.5412482 | 2.599469 | 0.00034239102 |
| hopping | 16 | dual_columns | 0.98065911 | 0.7207806 | 1.6827584 | 0.00034766461 |
| hopping | 16 | canonical_Lambda_k8 | 0.61790168 | 19.57205 | 3.1086668 | 0.00038448534 |
| hopping | 16 | canonical_Lambda_k16 | 0.76703607 | 11.566565 | 2.6744359 | 0.00035605101 |
| hopping | 16 | canonical_Lambda_k32 | 0.84974529 | 7.0355034 | 2.2914306 | 0.00034625144 |
| hopping | 16 | adaptive_energy_columns | 0.99999606 | 0.00016689422 | 2.0392093 | 7.2035533 |
| hopping | 32 | two_orbit | 0.99958693 | 0.021390134 | 1.7147801 | 0.00052984735 |
| hopping | 32 | physics_prior | 0.99958693 | 0.021390134 | 3.8439643 | 0.00051391873 |
| hopping | 32 | dual_columns | 0.99774177 | 0.10400378 | 1.7009165 | 0.00059723641 |
| hopping | 32 | canonical_Lambda_k8 | 0.79329017 | 10.40752 | 3.6850361 | 0.00061635227 |
| hopping | 32 | canonical_Lambda_k16 | 0.88263061 | 5.7415411 | 2.9470775 | 0.00060856127 |
| hopping | 32 | canonical_Lambda_k32 | 0.93203046 | 3.1715728 | 2.455658 | 0.00058878394 |
| hopping | 32 | adaptive_energy_columns | 1 | 6.4249545e-08 | 2.0451437 | 7.2035533 |
| interaction | 16 | two_orbit | 0.99998286 | 0.0011285798 | 1.7147801 | 0.00047435256 |
| interaction | 16 | physics_prior | 0.95586045 | 1.5199829 | 2.5998692 | 0.00035756907 |
| interaction | 16 | dual_columns | 0.98102112 | 0.69848415 | 1.6823694 | 0.00035057594 |
| interaction | 16 | canonical_Lambda_k8 | 0.61835203 | 19.553043 | 3.1092358 | 0.00038058694 |
| interaction | 16 | canonical_Lambda_k16 | 0.76701936 | 11.566386 | 2.6746713 | 0.00035794724 |
| interaction | 16 | canonical_Lambda_k32 | 0.84990831 | 7.0153044 | 2.2914799 | 0.00034279808 |
| interaction | 16 | adaptive_energy_columns | 0.9999987 | 4.8815953e-05 | 2.0253449 | 7.2222655 |
| interaction | 32 | two_orbit | 0.99998499 | 0.00094820136 | 1.7175513 | 0.00055203817 |
| interaction | 32 | physics_prior | 0.99999715 | 0.00018916448 | 3.8573195 | 0.00054149816 |
| interaction | 32 | dual_columns | 0.99813443 | 0.083224731 | 1.6998356 | 0.00059934484 |
| interaction | 32 | canonical_Lambda_k8 | 0.79376727 | 10.391963 | 3.6842304 | 0.00060841689 |
| interaction | 32 | canonical_Lambda_k16 | 0.88272538 | 5.7242821 | 2.9461159 | 0.00060156829 |
| interaction | 32 | canonical_Lambda_k32 | 0.93217964 | 3.1597255 | 2.4555253 | 0.00058685057 |
| interaction | 32 | adaptive_energy_columns | 1 | 1.5418574e-08 | 2.0284254 | 7.2222655 |

Matched-term contact comparisons (energy and cost jointly no worse; tolerance 1e-9):

| Competitor | Contact energy no worse | Contact energy and cost jointly no worse | Total paired tests |
|---|---:|---:|---:|
| physics_prior | 240 | 192 | 320 |
| dual_columns | 320 | 99 | 320 |
| canonical_Lambda_k8 | 320 | 320 | 320 |
| canonical_Lambda_k16 | 320 | 320 | 320 |
| canonical_Lambda_k32 | 320 | 320 | 320 |
| adaptive_energy_columns | 2 | 2 | 320 |

HFB is a separate 32-start, one-Gaussian baseline. Adaptive columns use the reused Gaussian overlap oracle on the energy Ritz residual, with no new exact target; their timing includes the full shared 32-column calculation and lists HFB setup separately in prediction metadata. Frozen column and canonical preparation costs are preserved in preregistration.json.

The contact construction has an untimed upstream discovery/proof cost, so these results do not establish an end-to-end computational advantage. Canonical sampling is the original coefficient measure Lambda, historically named P. All cost and actual-term differences must be retained when interpreting accuracy comparisons. Full paired cases and HFB results are in metrics.csv.
