"""All-particle Gaussian invariants and purity, without a dense operator stack."""
import numpy as np
from numba import njit, prange
from .tempering import actions


@njit(cache=True, parallel=True)
def _features(kets, permutations, phases, clifford):
    result = np.empty((len(kets), 4))
    errors = np.zeros(len(kets))
    for i in prange(len(kets)):
        cov = np.zeros((16, 16))
        p = 0
        for a in range(16):
            for b in range(a+1, 16):
                value = 0j
                for j in range(128):
                    value += kets[i, j].conjugate()*kets[i, permutations[p, j]]*phases[p, j]
                cov[a, b] = -value.imag
                cov[b, a] = value.imag
                p += 1
        ma, mb = np.zeros(7), np.zeros(7)
        for alpha in range(7):
            for a in range(8):
                for b in range(8):
                    ma[alpha] += clifford[alpha, a, b]*cov[a, b]/8
                    mb[alpha] += clifford[alpha, a, b]*cov[a+8, b+8]/8
        cross_norm = 0.
        for a in range(8):
            for b in range(8):
                cross_norm += cov[a, b+8]**2/8
        result[i, 0] = np.sum(ma*ma)
        result[i, 1] = np.sum(mb*mb)
        result[i, 2] = np.sum(ma*mb)
        result[i, 3] = cross_norm
        for a in range(16):
            for b in range(16):
                error = 1. if a == b else 0.
                for c in range(16):
                    error += cov[a, c]*cov[c, b]
                errors[i] = max(errors[i], abs(error))
    return result, errors


def full_distribution(kets, clifford):
    permutations, phases = actions(8)
    features, purity_errors = _features(np.ascontiguousarray(kets), permutations, phases,
                                       np.asarray(clifford, float))
    purity_max = float(purity_errors.max())
    if purity_max >= 1e-9:
        raise ArithmeticError("all-particle Gaussian purity gate failed")
    bins = np.linspace(0, 1, 33)
    hist, _, _ = np.histogram2d(features[:, 0], features[:, 3], bins=(bins, bins))
    return {"features": features, "purity_errors": purity_errors,
            "mass_a_eta_counts": hist, "bin_edges": bins}, {
                "feature_names": ["mass_a_squared", "mass_b_squared", "mass_dot", "eta"],
                "count": len(kets), "max_purity_error_all_particles": purity_max,
                "feature_means": features.mean(axis=0).tolist(),
                "samples_are_independent": False,
                "distribution_kind": "equal-weight resampled SMC approximation to canonical P_k"}

