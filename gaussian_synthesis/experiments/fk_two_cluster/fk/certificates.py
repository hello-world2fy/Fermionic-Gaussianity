"""Cost repairs and Schmidt-norm separation envelopes.

These formulae are rigorous for exact inputs.  Floating-point evaluations in
this pilot are explicitly numerical evidence, not interval certificates.
"""
from __future__ import annotations

import math
import numpy as np


def repaired_cost(coefficient_l1: float, residual: np.ndarray, *,
                  target_error: float = 0.0) -> float:
    """Repair an approximate representation using Gaussian Fock atoms.

``target_error`` bounds the 2-norm distance to the intended target.  The extra
sqrt(dim) factor repairs that error even when its coordinates are unavailable.
"""
    r = np.asarray(residual, dtype=complex).reshape(-1)
    if coefficient_l1 < 0 or target_error < 0:
        raise ValueError("Costs and error bounds must be nonnegative")
    if not np.isfinite(coefficient_l1) or not np.all(np.isfinite(r)):
        raise ValueError("Cost and residual must be finite")
    if not np.isfinite(target_error):
        raise ValueError("Target error must be finite")
    return float(coefficient_l1 + np.sum(np.abs(r)) + math.sqrt(r.size) * target_error)


def restricted_lower_bound(schmidt_norm: float, eta0: float = 0.25, *,
                           state_error: float = 0.0,
                           cluster_dimension: int = 16) -> float:
    """Lower bound for atoms with eta <= eta0, including target uncertainty.

For a 16 by 16 Schmidt matrix, |L(psi)-L(phi)| <= 4 ||psi-phi||_2.
"""
    if not 0 <= eta0 <= 1:
        raise ValueError("eta0 must lie in [0, 1]")
    if schmidt_norm < 0 or state_error < 0 or cluster_dimension < 1:
        raise ValueError("Norms and errors must be nonnegative; dimension positive")
    if not np.isfinite(schmidt_norm) or not np.isfinite(state_error):
        raise ValueError("Norms and errors must be finite")
    numerator = max(0.0, schmidt_norm - math.sqrt(cluster_dimension) * state_error)
    return float(numerator / (1.0 + math.sqrt(eta0)) ** 2)


def necessary_structure_bounds(schmidt_norm: float, cost_upper: float, *,
                               eta0: float = 0.25,
                               state_error: float = 0.0) -> dict:
    """Necessary coefficient-mass structure of ANY exact cost <= U expansion.

Use L_lower = max(L - 4 epsilon, 0), not the central Schmidt norm. The first
bound follows from L(psi) <= B (1 + sqrt(E eta))**2. The second uses
L(g) <= (1+sqrt(eta0))**2 for the low-eta part and L(g) <= 4 elsewhere.
These are coefficient absolute-mass fractions, not coherent probabilities.
"""
    lower_schmidt = restricted_lower_bound(schmidt_norm, 0.0, state_error=state_error)
    if cost_upper < 0 or not np.isfinite(cost_upper):
        raise ValueError("Cost upper bound must be finite and nonnegative")
    if not 0 <= eta0 <= 1:
        raise ValueError("eta0 must lie in [0, 1]")
    if cost_upper == 0 and lower_schmidt > 0:
        raise ValueError("A positive Schmidt norm cannot have zero representation cost")
    ratio = lower_schmidt / cost_upper if cost_upper else 0.0
    restricted_atom_cap = (1.0 + math.sqrt(eta0)) ** 2
    mean_eta = max(math.sqrt(ratio) - 1.0, 0.0) ** 2
    high_eta_mass = (max((ratio - restricted_atom_cap) / (4.0 - restricted_atom_cap), 0.0)
                     if eta0 < 1.0 else 0.0)
    return {
        "structure_cost_cap": float(cost_upper),
        "schmidt_norm_lower": float(lower_schmidt),
        "mean_eta_necessary_lower": float(mean_eta),
        "eta_above_threshold_mass_fraction_lower": float(high_eta_mass),
        "structure_bounds_scope": (
            "Any exact Gaussian decomposition with coefficient absolute-mass cost "
            "B <= structure_cost_cap; the mass fraction counts atoms with eta > eta0."
        ),
    }


def separation_certificate(coefficient_l1: float, residual: np.ndarray,
                           schmidt_norm: float, *, eta0: float = 0.25,
                           state_error: float = 0.0,
                           numerical_margin: float = 1e-8) -> dict:
    """Compare repaired full-dictionary cost with a restricted lower bound.

The numerical margin is an explicit floating-point cushion, not an interval
rounding bound.  Consequently this routine never claims a formal proof.
"""
    if numerical_margin < 0 or not np.isfinite(numerical_margin):
        raise ValueError("Numerical margin must be finite and nonnegative")
    upper = repaired_cost(coefficient_l1, residual, target_error=state_error)
    lower = restricted_lower_bound(schmidt_norm, eta0, state_error=state_error)
    structure = necessary_structure_bounds(schmidt_norm, upper, eta0=eta0,
                                           state_error=state_error)
    passed = upper + numerical_margin < lower - numerical_margin
    return {
        "eta0": float(eta0),
        "coefficient_l1": float(coefficient_l1),
        "fock_residual_l1": float(np.sum(np.abs(residual))),
        "residual_l2": float(np.linalg.norm(residual)),
        "state_error": float(state_error),
        "numerical_margin": float(numerical_margin),
        "numerical_margin_interpretation": (
            "A conservative numerical cushion for the comparison, not an interval "
            "rounding bound or a proven enclosure of all floating-point errors."
        ),
        "cost_upper": upper,
        "restricted_cost_lower": lower,
        "separation_gap": float(lower - upper),
        "numerical_separation": bool(passed),
        "evidence_level": "numerical_evidence" if passed else "not_separated",
        "strict_interval_certified": False,
        "statistically_certified": False,
        **structure,
        "interpretation": (
            "Exact-input inequalities separate costs; evaluated with floating-point "
            "atoms, symmetry projection, coefficients, and target. No directed "
            "rounding or interval eigenvector enclosure has been performed."
        ),
    }
