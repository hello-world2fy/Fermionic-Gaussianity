"""Quadratic onsite-PBCS constraints and the two-copy fidelity upper bound."""

from itertools import combinations
import numpy as np
from gaussian_synthesis.metrics import normalized


def minor_rows(basis, *, max_entries=2_000_000):
    D = basis.dimension
    index = {mask: i for i, mask in enumerate(basis.pair_masks)}
    rows = []
    if basis.m < 2 or basis.m > basis.L-2:
        return np.empty((0, D*D))
    for four in combinations(range(basis.L), 4):
        remaining = [j for j in range(basis.L) if j not in four]
        for rest in combinations(remaining, basis.m-2):
            R = sum(1 << j for j in rest)
            i, j, k, l = four
            def loc(a, b):
                return index[R | (1 << a) | (1 << b)]
            a, b = loc(i, j), loc(k, l)
            for c, d in [(loc(i, k), loc(j, l)), (loc(i, l), loc(j, k))]:
                if (len(rows)+1)*D*D > max_entries:
                    raise ValueError('Minor matrix exceeds the memory budget')
                row = np.zeros(D*D)
                row[a*D+b] = row[b*D+a] = .5
                row[c*D+d] = row[d*D+c] = -.5
                rows.append(row)
    return np.asarray(rows)


def minor_certificate(basis, state):
    state = normalized(basis.sector.vector(state, nonzero=True))
    rows = minor_rows(basis)
    if not len(rows):
        return {'upper_bound': 1., 'q': 0., 'max_minor': 0., 'rank': 0}
    _, singular, vh = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*max(rows.shape)*np.finfo(float).eps))
    doubled = np.kron(state, state)
    q = float(np.linalg.norm(vh[:rank] @ doubled)**2)
    return {'upper_bound': float(np.sqrt(max(0., 1-q))), 'q': q,
            'max_minor': float(np.max(np.abs(rows @ doubled))), 'rank': rank}
