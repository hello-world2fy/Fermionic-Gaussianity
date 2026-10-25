"""Low-cost Gaussian mass, spectator invariance, and configuration certificates.

Finite-order canonical mass is not the Gaussian atomic norm. For a normalized
Gaussian target the integer-k witness is exact, while the coefficient mass of
that continuous decomposition is the blank value B_k below. A normalized state
always has atomic mass at least 1 because |<psi|g>| <= 1.

Constructions stored as symmetry-orbit vectors P_sector|g> keep the feasible
mass sum |c_a|. They are not unprojected Gaussian atoms. Random overlaps give
lower bounds on a maximum, so they are not used as the constant c in the
necessity certificate.
"""
import heapq
import math
from fractions import Fraction
from itertools import combinations
import numpy as np
from scipy.optimize import minimize
from .model import PairModel
from .pairing_channels import PairKernelModel


def log_blank_coefficient_mass(n, k):
    """log B_k(g0) = sum_{i<j} log(2 a_ij + 2k) - log(2 a_ij + 2k - 1)."""
    if n < 1 or k < 1:
        raise ValueError('n and k must be positive')
    terms = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            a = 2 * n - i - j
            terms.append(math.log(2 * a + 2 * k) - math.log(2 * a + 2 * k - 1))
    return math.fsum(terms)


def blank_coefficient_mass(n, k):
    """Exact rational B_k for one normalized Gaussian on n modes."""
    value = Fraction(1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            a = 2 * n - i - j
            value *= Fraction(2 * a + 2 * k, 2 * a + 2 * k - 1)
    return value


def log_frame_dimension(n, s):
    """log d_{n,s} for real s >= 0, with d_{n,s} = prod_{i<j} (1 + s / a_ij)."""
    terms = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            a = 2 * n - i - j
            terms.append(math.log(a + s) - math.log(a))
    return math.fsum(terms)


def canonical_mean_fidelity(n, k):
    """Mean |<g0|g>|^2 of one atom drawn from the canonical coefficient law."""
    return math.exp(log_frame_dimension(n, k - 0.5) - log_frame_dimension(n, k + 0.5))


def minimal_k_within_mass(n, limit=1.1):
    """Smallest positive integer k with B_k(g0) <= limit. Float64 log search."""
    target = math.log(limit)
    if log_blank_coefficient_mass(n, 1) <= target:
        return 1
    hi = 1
    while log_blank_coefficient_mass(n, hi) > target:
        hi *= 2
        if hi > 1_000_000:
            raise RuntimeError('blank mass did not reach the requested limit')
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if log_blank_coefficient_mass(n, mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def quartic_tilt_energies(levels):
    """Odd-plus-even deformation that is not the earlier cubic or quadratic kick."""
    x = 2 * (np.arange(levels) + 0.5) / levels - 1
    raw = np.sort(x + 0.45 * x ** 4 + 0.18 * x ** 3)
    raw = raw - raw.mean()
    return raw / np.max(np.abs(raw))


def edge_bunched_energies(levels):
    """Levels concentrated at the band edges, with a gap around zero."""
    u = (np.arange(levels) + 0.5) / levels
    raw = np.sign(u - 0.5) * np.abs(2 * u - 1) ** 0.45
    raw = raw - raw.mean()
    return raw / np.max(np.abs(raw))


def all_to_all(energies, coupling):
    """Reduced BCS kernel g = G/L on every pair level, including the diagonal."""
    energies = np.asarray(energies, float)
    if np.any(np.diff(energies) < -1e-12):
        raise ValueError('energies must be nondecreasing')
    levels = len(energies)
    kernel = np.full((levels, levels), coupling / levels)
    return PairKernelModel(energies, kernel)


def spectator_extension(energies, coupling, gap=4.0):
    """Add one filled and one empty pair level with zero pairing kernel.

    The active coupling stays G/L_active. True Gaussian mass is invariant when
    these levels remain in a pure Slater state. Finite-order blank mass is not,
    because it depends on the total mode count.
    """
    active = np.asarray(energies, float)
    filled = float(active.min() - gap)
    empty = float(active.max() + gap)
    extended = np.r_[filled, active, empty]
    levels = len(extended)
    kernel = np.zeros((levels, levels))
    kernel[1:-1, 1:-1] = coupling / len(active)
    return PairKernelModel(extended, kernel)


def embed_active_state(active, extended, state, n_filled=1):
    """Place `state` into the sector where the filled spectators stay occupied."""
    if extended.L != active.L + n_filled + 1:
        raise ValueError('this embedding expects n_filled occupied spectators and one empty spectator')
    out = np.zeros(extended.dim, complex)
    occupied = (1 << n_filled) - 1
    for amplitude, mask in zip(state, active.masks):
        big = occupied | (int(mask) << n_filled)
        out[extended.index[big]] = amplitude
    return out


def outside_configuration_fraction(witness_overlap, witness_norm, error, budget, class_overlap):
    """Lower bound on the coefficient mass sitting outside a configuration class.

    The witness must already satisfy sup_G |<y|g>| <= 1. `class_overlap` must
    be an upper bound on sup_D |<y|g>|, not a random-search maximum. Returns
    None when that upper bound is not strictly below 1.
    """
    if not math.isfinite(class_overlap) or class_overlap >= 1:
        return None
    numerator = (witness_overlap - witness_norm * error) / budget - class_overlap
    return float(max(0.0, numerator) / (1 - class_overlap))


def trivial_mass_lower_bound(state):
    """B >= 1 for a normalized state, since every pure Gaussian has overlap <= 1."""
    norm = float(np.linalg.norm(state))
    return {'mass_lower': 1.0 if abs(norm - 1) < 1e-10 else None,
            'norm': norm,
            'reason': '|<psi|g>| <= ||g|| = 1 and |<psi|psi>| = 1 imply atomic mass >= 1',
            'arithmetic': 'float64, no interval enclosure'}


def _cos_range(left, right):
    values = [math.cos(left), math.cos(right)]
    start = math.ceil(left / (2 * math.pi))
    stop = math.floor(right / (2 * math.pi))
    for k in range(start, stop + 1):
        values.append(math.cos(2 * math.pi * k))
    start = math.ceil((left - math.pi) / (2 * math.pi))
    stop = math.floor((right - math.pi) / (2 * math.pi))
    for k in range(start, stop + 1):
        values.append(math.cos(math.pi + 2 * math.pi * k))
    return min(values), max(values)


def _annulus_box(radius_min, radius_max, angle_left, angle_right):
    cosine = _cos_range(angle_left, angle_right)
    sine = _cos_range(angle_left - math.pi / 2, angle_right - math.pi / 2)
    def axis(interval):
        samples = [radius * trig for radius in (radius_min, radius_max) for trig in interval]
        return min(samples), max(samples)
    real = axis(cosine)
    imag = axis(sine)
    return (real[0], real[1], imag[0], imag[1])


def _mul_box(left, right):
    corners = []
    for ar in (left[0], left[1]):
        for ai in (left[2], left[3]):
            for br in (right[0], right[1]):
                for bi in (right[2], right[3]):
                    corners.append((ar * br - ai * bi, ar * bi + ai * br))
    return (min(z[0] for z in corners), max(z[0] for z in corners),
            min(z[1] for z in corners), max(z[1] for z in corners))


def _add_box(left, right):
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2], left[3] + right[3])


def _box_modulus(box):
    corners = (box[0], box[1], box[2], box[3])
    return max(math.hypot(re, im) for re in corners[:2] for im in corners[2:])


def product_gaussian_overlap(amplitudes, thetas, phases):
    """Overlap of a seniority-zero vector with an onsite product Gaussian.

    Level j contributes cos(theta_j)|empty> + sin(theta_j) exp(i phase_j)|pair>.
    This is the full-Fock overlap, because the target has no broken-pair weight.
    """
    levels = len(thetas)
    pairs = levels // 2
    total = 0j
    for index, occupied in enumerate(combinations(range(levels), pairs)):
        amplitude = 1 + 0j
        chosen = set(occupied)
        for level in range(levels):
            if level in chosen:
                amplitude *= math.sin(thetas[level]) * np.exp(1j * phases[level])
            else:
                amplitude *= math.cos(thetas[level])
        total += np.conjugate(amplitudes[index]) * amplitude
    return complex(total)


def _product_overlap_upper(amplitudes, boxes):
    """Interval upper bound. Each box is (theta_lo, theta_hi, phase_lo, phase_hi)."""
    amplitudes = np.asarray(amplitudes, complex)
    levels = len(boxes)
    pairs = levels // 2
    factors = []
    for theta_lo, theta_hi, phase_lo, phase_hi in boxes:
        empty = (math.cos(theta_hi), math.cos(theta_lo), 0.0, 0.0)
        occupied = _annulus_box(math.sin(theta_lo), math.sin(theta_hi), phase_lo, phase_hi)
        factors.append((empty, occupied))
    total = (0.0, 0.0, 0.0, 0.0)
    for index, occupied in enumerate(combinations(range(levels), pairs)):
        chosen = set(occupied)
        # <psi|g> multiplies the Gaussian amplitude by the conjugate coefficient.
        coefficient = np.conjugate(amplitudes[index])
        term = (float(coefficient.real), float(coefficient.real),
                float(coefficient.imag), float(coefficient.imag))
        for level in range(levels):
            term = _mul_box(term, factors[level][1 if level in chosen else 0])
        total = _add_box(total, term)
    # A tiny outward pad keeps the float64 rectangle bound from rounding below the truth.
    return _box_modulus(total)*(1+1e-12)+1e-15


def _half_filled_levels(dimension):
    pairs = 0
    while pairs < 8 and math.comb(2 * pairs, pairs) != dimension:
        pairs += 1
    if math.comb(2 * pairs, pairs) != dimension:
        raise ValueError('branch-and-bound certificate is implemented for half-filled pair states')
    return 2 * pairs


def _local_product_lower(amplitudes, starts, seed):
    levels = _half_filled_levels(len(amplitudes))
    rng = np.random.default_rng(seed)

    def objective(x):
        thetas = x[:levels]
        phases = np.r_[0.0, x[levels:]]
        return -abs(product_gaussian_overlap(amplitudes, thetas, phases))

    best = 0.0
    for k in range(starts):
        guess = np.r_[np.full(levels, math.pi / 4), np.zeros(levels - 1)]
        if k:
            guess = np.r_[rng.uniform(0, math.pi / 2, levels), rng.uniform(0, 2 * math.pi, levels - 1)]
        fit = minimize(objective, guess, method='L-BFGS-B',
                       bounds=[(0, math.pi / 2)] * levels + [(0, 2 * math.pi)] * (levels - 1),
                       options={'maxiter': 80, 'ftol': 1e-12})
        best = max(best, float(-fit.fun))
    return best


def certified_onsite_overlap(amplitudes, *, tolerance=0.02, max_boxes=40000, starts=8, seed=7, region=None):
    """Global upper and lower bounds on sup |<y|g>| over onsite product Gaussians.

    The upper bound comes from interval arithmetic on parameter boxes. A local
    search only initializes the lower bound. If the gap stays above `tolerance`,
    the returned upper bound is still valid and the flag says it did not close.
    Implemented for four pair levels, where the product has seven free angles.
    """
    amplitudes = np.asarray(amplitudes, complex)
    if len(amplitudes) != 6:
        raise ValueError('certified onsite bound currently covers four pair levels')
    lower = _local_product_lower(amplitudes, starts, seed)
    # Also evaluate the equal-amplitude chart; it is often near the maximum.
    equal = abs(product_gaussian_overlap(amplitudes, np.full(4, math.pi / 4), np.zeros(4)))
    lower = max(lower, float(equal))
    if region is None:
        initial = tuple((0.0, math.pi / 2, 0.0, 2 * math.pi) for _ in range(4))
        # Gauge-fix level 0. Its phase box is a point, which the interval still accepts.
        initial = ((0.0, math.pi / 2, 0.0, 0.0),) + initial[1:]
    else:
        initial = tuple(tuple(map(float, box)) for box in region)
        if len(initial) != 4:
            raise ValueError('onsite region must contain four levels')
    heap = []
    upper = _product_overlap_upper(amplitudes, initial)
    heapq.heappush(heap, (-upper, 0, initial))
    processed = 0
    closed = upper <= lower + tolerance
    while heap and not closed and processed < max_boxes:
        neg, _, boxes = heapq.heappop(heap)
        box_upper = -neg
        if box_upper <= lower + tolerance:
            closed = True
            break
        widths = []
        for theta_lo, theta_hi, phase_lo, phase_hi in boxes:
            widths.append(theta_hi - theta_lo)
            widths.append(phase_hi - phase_lo)
        split = int(np.argmax(widths))
        level, kind = divmod(split, 2)
        left = list(boxes)
        right = list(boxes)
        theta_lo, theta_hi, phase_lo, phase_hi = boxes[level]
        if kind == 0:
            mid = 0.5 * (theta_lo + theta_hi)
            left[level] = (theta_lo, mid, phase_lo, phase_hi)
            right[level] = (mid, theta_hi, phase_lo, phase_hi)
        else:
            mid = 0.5 * (phase_lo + phase_hi)
            left[level] = (theta_lo, theta_hi, phase_lo, mid)
            right[level] = (theta_lo, theta_hi, mid, phase_hi)
        for child in (tuple(left), tuple(right)):
            child_upper = _product_overlap_upper(amplitudes, child)
            center_theta = [0.5 * (box[0] + box[1]) for box in child]
            center_phase = [0.5 * (box[2] + box[3]) for box in child]
            lower = max(lower, abs(product_gaussian_overlap(amplitudes, center_theta, center_phase)))
            if child_upper > lower + tolerance:
                heapq.heappush(heap, (-child_upper, processed, child))
        processed += 1
        if heap:
            closed = -heap[0][0] <= lower + tolerance
    certified_upper = -heap[0][0] if heap else lower
    return {'overlap_lower': float(lower), 'overlap_upper': float(certified_upper),
            'gap': float(certified_upper - lower), 'closed': bool(certified_upper <= lower + tolerance),
            'boxes_processed': processed, 'tolerance': tolerance,
            'global_upper_bound': True,
            'random_search_is_not_the_upper_bound': True,
            'family': 'onsite product Gaussians, four pair levels',
            'arithmetic': 'float64 interval bounds, no directed rounding'}
