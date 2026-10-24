# Number-conserving pairing: a second physical mechanism test

The selected four-orbital pairing model has a rigorously unique Gaussian minimum (the filled two-lowest-pair Slater state) at all three couplings. Global Gaussian coercivity, full-Fock spectral separation, positive target amplitudes and finite window constructions were independently checked.

| Pair coupling g | Required eighth-excitation amplitude q8 | Full spectral-gap lower | Gaussian coercivity c |
|---|---:|---:|---:|
| 0.05 | 9.24313000156e-05 | 0.8 | 0.775 |
| 0.1 | 0.000374714560626 | 0.6 | 0.55 |
| 0.2 | 0.00153645014688 | 0.2 | 0.1 |

For exact representation, the coefficient cost is Theta(Delta^-2). The lower bound applies to every Gaussian in the energy window. Eighteen actual normalized Gaussians give a matching-power upper bound at every sufficiently small Delta; twelve concrete constructions (three g values and four windows) were verified in full Fock space. No optimal leading constant is claimed.

At the fixed Hilbert error epsilon=q8, dropping the eighth-excitation component yields a 13-Gaussian construction of cost O(Delta^-1). A fourth-excitation functional gives a matching Omega(Delta^-1) lower bound because q4>q8. For every fixed epsilon<q8, the eighth-excitation obstruction still forces Delta^-2. The finite-error approximation is generally unnormalized; this is a representation-cost theorem, not a normalized energy-accuracy claim.

This is a different number-conserving Hamiltonian and a global Gaussian-dictionary result. It tests the energy-curvature/excitation-order mechanism beyond the FK singlet geometry. The alternative three-cluster FK candidate was prepared as a 4096-dimensional sparse model, but was not selected for an ED-only scale-up.
