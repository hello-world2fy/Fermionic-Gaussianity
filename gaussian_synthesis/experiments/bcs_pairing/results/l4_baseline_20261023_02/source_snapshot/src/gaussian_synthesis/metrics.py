"""Model-independent pure-state comparisons with explicit norm handling."""

import numpy as np


def normalized(vector):
    vector = np.array(vector, dtype=np.complex128, copy=True)
    if vector.ndim != 1 or not len(vector) or not np.all(np.isfinite(vector)):
        raise ValueError("Expected a nonempty finite vector")
    scale = np.max(np.abs(vector))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Cannot normalize a zero or nonfinite vector")
    vector /= scale
    return vector / np.linalg.norm(vector)


def _pair(left, right):
    left, right = normalized(left), normalized(right)
    if left.shape != right.shape:
        raise ValueError("State vectors must have matching dimensions")
    return left, right


def fidelity(left, right):
    """Squared overlap of normalized rays; inputs may have arbitrary nonzero norms."""
    left, right = _pair(left, right)
    return float(np.clip(abs(np.vdot(left, right))**2, 0., 1.))


def phase_aligned_distance(left, right):
    """Euclidean distance of normalized rays after optimal common-phase alignment."""
    left, right = _pair(left, right)
    overlap = np.vdot(left, right)
    phase = overlap.conjugate()/abs(overlap) if abs(overlap) else 1.
    return float(np.linalg.norm(left-phase*right))
