"""Directed interval evaluation of the proven order-two Gaussian projector.

This deliberately independent route certifies a full-dictionary witness even
when a high-order Monte Carlo normalization cannot be certified. It is NOT a
certificate for the stronger high-order point estimate. The exact witness is
specified by binary64 coefficients in the explicit radical FK singlet basis.
The supplied target is the exact vector defined by its saved binary64 entries;
comparison with an exact Hamiltonian eigenvector needs a separate state-error
bound and is never silently assumed here.
"""
import math
import numpy as np
from mpmath import iv


def _c(z):
    return iv.mpf(float(np.real(z))), iv.mpf(float(np.imag(z)))


def _mul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def _scale(a, x):
    return a[0]*x, a[1]*x


def _bounds(x):
    # Conversion to binary64 is nearest rounding; expand one ULP outward.
    return [float(np.nextafter(float(x.a), -np.inf)), float(np.nextafter(float(x.b), np.inf))]


def interval_q2(z, digits=70):
    iv.dps = digits
    z = [_c(x) for x in z]
    v = [_scale(_mul(z[i], z[j]), iv.sqrt(2) if i != j else 1)
         for i in range(3) for j in range(i, 3)]
    entries = [[99, 0, 0, 1, 0, 3], [0, 100, 0, 0, 4*iv.sqrt(7), 0],
               [0, 0, 96, 4*iv.sqrt(14), 0, 0], [1, 0, 4*iv.sqrt(14), 99, 0, 25],
               [0, 4*iv.sqrt(7), 0, 0, 124, 0], [3, 0, 0, 25, 0, 99]]
    result = iv.mpf(0)
    for i in range(6):
        for j in range(6):
            result += entries[i][j]*(v[i][0]*v[j][0]+v[i][1]*v[j][1])/128
    return result


def exact_basis_action(z):
    """The explicit radical S from the existing signed-integer projector proof."""
    masks = [m for m in range(256) if m.bit_count()%2 == 0]
    indices = {m: i for i, m in enumerate(masks)}
    supports = [((119, 1), (120, -1), (135, -1), (136, 1)),
                ((15, -1), (60, 1), (90, -1), (105, 1), (150, 1), (165, -1), (195, 1), (240, -1)),
                ((30, 2), (45, -2), (75, 2), (180, 2), (210, -2), (225, 2),
                 (119, -1), (120, -1), (135, -1), (136, -1))]
    scalars = [(iv.mpf(1)/2, iv.mpf(0)), (iv.mpf(0), 1/iv.sqrt(8)),
               (1/(2*iv.sqrt(7)), iv.mpf(0))]
    output = [(iv.mpf(0), iv.mpf(0)) for _ in masks]
    for j in range(3):
        value = _mul(_c(z[j]), scalars[j])
        for mask, integer in supports[j]:
            i = indices[mask]
            output[i] = (output[i][0]+integer*value[0], output[i][1]+integer*value[1])
    return output


def certify_order2_normalization(z, target, digits=70):
    z = np.asarray(z, complex)
    target = np.asarray(target, complex)
    if z.shape != (3,) or target.shape != (128,) or not np.isfinite(z).all() or not np.isfinite(target).all():
        raise ValueError("finite three-coordinate witness and full even target required")
    q_interval = interval_q2(z, digits)
    q_bounds = _bounds(q_interval)
    if q_bounds[0] <= 0:
        raise ValueError("positive q2 required")
    n_upper = _bounds(q_interval**(iv.mpf(1)/4))[1]
    z_safe = z/(n_upper*(1+32*np.finfo(float).eps))
    normalized_bounds = _bounds(interval_q2(z_safe, digits))
    if normalized_bounds[1] > 1:
        raise ArithmeticError("interval witness feasibility check failed")
    exact_witness = exact_basis_action(z_safe)
    inner = iv.mpf(0)
    norm2 = iv.mpf(0)
    for (wr, wi), t in zip(exact_witness, target):
        tr, ti = _c(t)
        inner += wr*tr+wi*ti
        norm2 += wr*wr+wi*wi
    return {"z_feasible": z_safe,
            "q2_input_interval": q_bounds, "q2_normalized_interval": normalized_bounds,
            "L_certified_saved_target": _bounds(inner)[0],
            "witness_hilbert_norm_upper": _bounds(iv.sqrt(norm2))[1],
            "gaussian_support_upper": 1., "interval_decimal_digits": digits,
            "full_dictionary_feasible": True,
            "witness_specification": "Exact radical S multiplied by saved binary64 z_feasible; displayed full ket is approximate",
            "target_scope": "Exact vector defined by the supplied binary64 Fock amplitudes; not an interval-certified Hamiltonian eigenstate",
            "normalization_order": 2, "high_order_moment_certified": False,
            "proof": "h_G(w)^4 <= <w tensor w|P_G|w tensor w> = q2(w) <= 1; exact closed-form matrix verified by predecessor signed-integer proof"}

