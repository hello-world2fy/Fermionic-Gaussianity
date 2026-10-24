# Normalized energy-window bounds and actual variational states

All 48 global bounds passed independent 110-digit and exact-rational checks. All 96 saved trial states satisfy normalization and coefficient budgets. The 12 dictionaries contain 1164 actual Gaussian atoms checked in full Fock space.

There are 13 recorded non-optimal solver status entries (including repeated subproblems). Numerical trial states are feasible upper benchmarks; no global energy optimum is claimed. Budgets use the original plan reference U.

At Delta=0.01 and budget U:

| t | Feasible fidelity | Global fidelity upper | Global energy-error lower | Feasible energy error |
|---|---:|---:|---:|---:|
| 9 | 0.4892500010 | 0.4983855394 | 11.1821256270 | 11.3886880656 |
| 11 | 0.6410559555 | 0.6414123527 | 8.3275579651 | 8.3359681892 |
| 20 | 0.9435684852 | 0.9435800031 | 2.7357988108 | 2.7363587452 |

The global bound applies to any number of Gaussian terms. The small remaining gap at t11 and t20 is a bound on the room for improvement by any energy-window ansatz under the same budget, not just this 97-atom dictionary.

D3 independently checks 36 finite FK constructions (three couplings, four windows, three errors), with matching divergence powers, and four five-Gaussian Fourier constructions attaining the toy leading constant. Positive-error FK approximants are not required to be normalized.
