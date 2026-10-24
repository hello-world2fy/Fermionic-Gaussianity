# Frozen high-order directions and the finite-order stopping rule

All eight frozen directions passed independent 110-digit checks of the exact q4, global support envelope, explicit Gaussian support lower witness, physical target overlap and rigorous high-order moment envelopes. The high-order point integrals themselves remain uncertified.

| t | k | Certified cost lower | Best possible cost lower from this direction | Joint baseline |
|---|---|---:|---:|---:|
| 11 | 8 | 1.409751637 | <= 1.576783573 | 1.672686220 |
| 11 | 16 | 1.344367464 | <= 1.654136418 | 1.672686220 |
| 11 | 32 | 1.208568704 | <= 1.690009529 | 1.672686220 |
| 11 | 64 | 1.077560751 | <= 1.713788855 | 1.672686220 |
| 20 | 8 | 1.116384760 | <= 1.129389167 | 1.287430340 |
| 20 | 16 | 1.131520473 | <= 1.193250057 | 1.287430340 |
| 20 | 32 | 1.013202147 | <= 1.251870948 | 1.287430340 |
| 20 | 64 | 0.850504984 | <= 1.284120387 | 1.287430340 |

All four t20 directions, and t11 k8/k16, cannot beat the analytic joint witness even with a perfect global support computation. The t11 k32/k64 directions remain possible candidates, but current global bounds do not certify an improvement. No unconditional increase of k is warranted.

For t20 the scaled main/secondary log-height ratios at k8,16,32,64 are about 41.54,63.70,75.81,60.50, while the proposed two-orbit balance requires about 5.105. The current sequence is not in that asymptotic regime. Thresholding eta is only a finite-width diagnostic, not a certified assignment to contact components. The known-optimum toy demonstrates 1/k drift under its proved hypotheses; those hypotheses cannot be transferred to the unclassified FK optimal witness.

The deterministic high-order bounds use Cartan block inclusion and Jensen. They are wide envelopes, not validated high-dimensional quadrature. The toy four-radial reduction does not apply to a general FK singlet direction: the full real Gaussian manifold has dimension 56, and quotienting by diagonal Spin(7) (dimension 21) leaves at least 35 generic dimensions.
