"""Fit-independent constraints for two-density Jastrow amplitudes.

Two disjoint interband swaps and one intraband swap define an eight-vertex
Boolean cube at fixed total number. Every quadratic density exponent and every
arbitrary function of transfer count has zero alternating third difference.
Consequently products of four even and four odd vertex probabilities agree.
This is an algebraic diagnostic, not a claim of new statistical theory.
"""
from itertools import combinations, product
import numpy as np

EVEN = np.array([0, 3, 5, 6])
ODD = np.array([1, 2, 4, 7])


def mixed_swap_cubes(model):
    """Enumerate cubes for L<=12; large-L exhaustive enumeration is expensive."""
    if model.L > 12:
        raise ValueError('Use a preselected subset of swaps for larger systems')
    all_rows = []
    bits = np.array(list(product((0, 1), repeat=3)), int)
    lookup = np.full(1 << model.L, -1, int)
    lookup[model.masks] = np.arange(model.dim)
    for h in combinations(range(model.r), 2):
        for p in combinations(range(model.r, model.L), 2):
            for particle in (p, p[::-1]):
                used = set(h+p)
                for band in (range(model.r), range(model.r, model.L)):
                    for intra in combinations([j for j in band if j not in used], 2):
                        pairs = np.array([(h[0], particle[0]), (h[1], particle[1]), intra])
                        selected = pairs[np.arange(3)[None], bits]
                        masks = np.sum(1 << selected, axis=1)
                        spectators = [j for j in range(model.L) if j not in used and j not in intra]
                        outside = np.array([sum(1 << j for j in s) for s in combinations(spectators, model.r-3)])
                        all_rows.append(lookup[outside[:, None] | masks[None, :]])
    return np.concatenate(all_rows).astype(np.int32)


def cube_diagnostics(probability, cubes):
    p = np.asarray(probability)[cubes]
    mass = p.sum(axis=1)
    root = np.sqrt(p/np.maximum(mass[:, None], 1e-300))
    invariant = np.prod(root[:, EVEN], axis=1)-np.prod(root[:, ODD], axis=1)
    # On the unit positive ball, ||grad(prod_even x-prod_odd x)||<=1/4.
    # Thus ||sqrt(p_cond)-sqrt(q_cond)||>=4|invariant|, and local
    # squared affinity <=(1-8 invariant^2)^2. Optimize candidate cube mass
    # using Cauchy-Schwarz to obtain an additive global infidelity lower bound.
    local_infidelity = 16*invariant**2-64*invariant**4
    return mass, invariant, mass*local_infidelity


def disjoint_cube_bound(probability, cubes):
    probability = np.asarray(probability, float)
    probability = probability/probability.sum()
    mass, invariant, contribution = cube_diagnostics(probability, cubes)
    occupied = np.zeros(len(probability), bool)
    chosen = []
    for index in np.argsort(contribution)[::-1]:
        rows = cubes[index]
        if contribution[index] <= 1e-30 or occupied[rows].any():
            continue
        occupied[rows] = True
        chosen.append(int(index))
    gap = float(contribution[chosen].sum())
    best = int(np.argmax(contribution))
    return {'fidelity_upper_bound_float64': 1.-gap,
            'infidelity_lower_bound_float64': gap,
            'interval_certified': False,
            'scope': 'Any two-body occupation Jastrow with arbitrary complex radial coefficients, including arbitrary phase relaxation; not general projected Gaussian or Gaussian sums',
            'candidate_cubes': len(cubes), 'disjoint_cubes': len(chosen),
            'covered_probability': float(mass[chosen].sum()),
            'largest_normalized_invariant': float(np.max(abs(invariant))),
            'best_cube': {'index': best, 'rows': cubes[best].tolist(), 'mass': float(mass[best]),
                          'invariant': float(invariant[best]), 'infidelity_contribution': float(contribution[best]),
                          'probabilities': probability[cubes[best]].tolist()},
            'chosen_indices': chosen}
