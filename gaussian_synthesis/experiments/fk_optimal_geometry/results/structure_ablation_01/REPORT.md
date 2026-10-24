# Four-coupling structure ablation

Completed and independently checked 720 configurations: four couplings, six dictionary sizes, five analytic/deletion/window dictionaries, and five independent replicas of each prior/column/canonical baseline. Each configuration has unconstrained fidelity and energy solutions plus normalized cost-budget fidelity and energy trials (2880 saved states).

The budget is the explicit original plan reference U at each coupling. Canonical pools are the coefficient measure Lambda proportional to |overlap|^(2k-1); legacy upstream filenames called this P. No claim that these are the overlap^(2k) Husimi distributions is made. Every pool is deduplicated before target-independent QR selection.

Numerical budget optimizations provide feasible candidates; the normalized fidelity relaxation may have an interior solution and energy optimization is local. The report retains the best feasible candidate from each nested smaller dictionary, so increasing the allowed term count cannot worsen the reported result. All original solver results and statuses remain saved.

Solver warnings: 233 entries including repeated subproblems. Nested candidate reuse: 187. All applicable whole-orbit and energy-window bounds are respected.

At R=16, means over the five repeats where applicable:

| t | Method | Budget fidelity | Budget energy error | Mass-correlation error (energy state) | Coefficient cost (energy state) |
|---|---|---:|---:|---:|---:|
| 9 | two_orbit | 0.994554562 | 0.37441065 | 1.0723514 | 1.88253129 |
| 9 | main_only | 0.975325254 | 1.6965555 | 2.148597 | 1.75463123 |
| 9 | secondary_only | 0.061049630 | 55.756273 | 9.1109056 | 1.59914065 |
| 9 | joint_contacts | 0.969700170 | 2.5922504 | 1.7887454 | 1.79808847 |
| 9 | low_energy_delta0.01 | 0.488325076 | 11.407412 | 1.3373253 | 1.88253128 |
| 9 | physics_prior | 0.939947688 | 3.5071278 | 3.3914065 | 1.88253129 |
| 9 | dual_columns | 0.979659735 | 0.7418069 | 0.55885901 | 1.88043953 |
| 9 | canonical_Lambda_k8 | 0.308648999 | 27.95808 | 3.6487179 | 1.88251906 |
| 9 | canonical_Lambda_k16 | 0.420215453 | 21.871422 | 2.552163 | 1.88252631 |
| 9 | canonical_Lambda_k32 | 0.625282754 | 13.921456 | 2.1830003 | 1.88252986 |
| 10.9225 | two_orbit | 0.999336836 | 0.055378002 | 0.27316384 | 1.75706807 |
| 10.9225 | main_only | 0.991614577 | 0.70023085 | 0.9091975 | 1.65864334 |
| 10.9225 | secondary_only | 0.025534903 | 63.262492 | 10.640098 | 1.65793681 |
| 10.9225 | joint_contacts | 0.990911480 | 0.88054782 | 0.77463185 | 1.68406662 |
| 10.9225 | low_energy_delta0.01 | 0.634749778 | 8.4501998 | 0.76969859 | 1.75706806 |
| 10.9225 | physics_prior | 0.964936980 | 2.0922861 | 1.5995767 | 1.71473830 |
| 10.9225 | dual_columns | 0.973610222 | 1.0358854 | 0.43934933 | 1.75587672 |
| 10.9225 | canonical_Lambda_k8 | 0.296737616 | 29.084441 | 5.00043 | 1.75706317 |
| 10.9225 | canonical_Lambda_k16 | 0.452950585 | 21.288389 | 3.6965293 | 1.75705087 |
| 10.9225 | canonical_Lambda_k32 | 0.604674617 | 14.705419 | 2.8870188 | 1.75706664 |
| 11 | two_orbit | 0.999407368 | 0.049880442 | 0.2543575 | 1.75145758 |
| 11 | main_only | 0.992009664 | 0.67252781 | 0.87414744 | 1.65431638 |
| 11 | secondary_only | 0.024551660 | 63.605211 | 10.682845 | 1.66044910 |
| 11 | joint_contacts | 0.991370677 | 0.84021896 | 0.74664582 | 1.67899880 |
| 11 | low_energy_delta0.01 | 0.640994908 | 8.3373205 | 0.82652843 | 1.75145758 |
| 11 | physics_prior | 0.964963312 | 2.0757844 | 1.5279699 | 1.75145758 |
| 11 | dual_columns | 0.977839053 | 0.92391593 | 0.38475936 | 1.74925219 |
| 11 | canonical_Lambda_k8 | 0.295469570 | 29.079414 | 4.9279275 | 1.75143930 |
| 11 | canonical_Lambda_k16 | 0.451992079 | 21.982913 | 3.9120185 | 1.75144343 |
| 11 | canonical_Lambda_k32 | 0.580293857 | 15.669254 | 2.8942174 | 1.75145736 |
| 20 | two_orbit | 1.000000000 | 0 | 1.2256045e-09 | 1.29557042 |
| 20 | main_only | 0.999963238 | 0.0059381055 | 0.0099090727 | 1.28788133 |
| 20 | secondary_only | 0.000339187 | 133.7594 | 8.214549 | 1.29741769 |
| 20 | joint_contacts | 0.999963172 | 0.0059294078 | 0.0095133747 | 1.28777428 |
| 20 | low_energy_delta0.01 | 0.943557580 | 2.7367609 | 1.8397838 | 1.29741769 |
| 20 | physics_prior | 0.997722761 | 0.19388613 | 0.12330042 | 1.26569836 |
| 20 | dual_columns | 0.998627689 | 0.12057947 | 0.0094313391 | 1.29424398 |
| 20 | canonical_Lambda_k8 | 0.231739481 | 47.026617 | 6.8280635 | 1.29741367 |
| 20 | canonical_Lambda_k16 | 0.372245020 | 35.67556 | 5.4518537 | 1.29741752 |
| 20 | canonical_Lambda_k32 | 0.563469812 | 22.652605 | 3.7826526 | 1.29741337 |

The two-orbit construction reaches the exact target with 16 actual Gaussians without a budget constraint. Its exact costs at t9,10.9225,11,20 are approximately 1.984228,1.793456,1.785866,1.295570. Only t20 lies within the corresponding reference U; the first three budget-constrained results remain inexact. Removing the secondary orbit leaves a provable singlet-amplitude obstruction, though the t20 fidelity loss is small. Restricting all atoms to the low-energy window gives the much stronger budget-dependent physical obstruction certified in D1/D2.

These are representation and structure comparisons on known FK targets. They do not establish a prediction advantage on new Hamiltonians; that claim requires the separately preregistered F tests. Unconstrained solutions and their potentially large coefficient costs are in metrics.csv.
