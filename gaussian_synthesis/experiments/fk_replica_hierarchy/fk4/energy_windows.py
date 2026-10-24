"""Full-dictionary energy-window obstruction and an explicit orbit construction.

For t>=9, every Gaussian has E-E_G >=18 x^2+4(t-9)x, with
x=1+<H_hop>/4. Relative to the hopping Gaussian vacuum, <N>=4x.
Gaussian canonical pairing then gives |<8|g>| <=(<N>/8)^2=x^2/4.
These yield a full-Gaussian restricted-cost lower bound. Three actual
Gaussian saturation-arc orbits inside the window give a constructive upper
bound with the same divergence exponent. No out-of-window Fock repair is
permitted or used for that construction.
"""
import math
import numpy as np
import mpmath as mp


def hopping_eigenbasis():
    """Columns have H_hop eigenvalues -4,0,+4 in the original S basis."""
    return np.array([[.25, math.sqrt(14)/4, .25],
                     [-math.sqrt(2)/2, 0, math.sqrt(2)/2],
                     [math.sqrt(7)/4, -math.sqrt(2)/4, math.sqrt(7)/4]])


def x_cap(t, delta):
    if t < 9 or delta <= 0:
        raise ValueError("t>=9 and positive window required")
    a = 4*(float(t)-9)
    return min(2., 2*float(delta)/(math.sqrt(a*a+72*float(delta))+a))


def arc_projection_x(x):
    return np.array([(1-x/2)**2, 3/(2*math.sqrt(14))*(2*x-x*x), x*x/4])


def window_construction(target_z, t, delta, *, error_allowance=0., digits=90):
    """High precision implementation of explicit coefficients, no ill-conditioned solve.

Exact coefficients are defined by the displayed rational/radical formulas.
Rounded decimal strings and their residual are saved, rather than treating
large-cancellation binary64 ket sums as exact reconstructions.
"""
    mp.mp.dps = digits
    z = [mp.mpc(float(v.real), float(v.imag)) for v in target_z]
    V = mp.matrix([[mp.mpf(1)/4, mp.sqrt(14)/4, mp.mpf(1)/4],
                   [-mp.sqrt(2)/2, 0, mp.sqrt(2)/2],
                   [mp.sqrt(7)/4, -mp.sqrt(2)/4, mp.sqrt(7)/4]])
    q = V.T*mp.matrix(z)
    d, coupling = mp.mpf(float(delta)), mp.mpf(float(t))
    aa = 4*(coupling-9)
    cap = min(mp.mpf(2), 2*d/(mp.sqrt(aa*aa+72*d)+aa))
    x = cap*mp.mpf("0.99")
    original_q = q.copy()
    if error_allowance and abs(q[2]) > 0:
        q[2] *= max(mp.mpf(0), 1-mp.mpf(float(error_allowance))/abs(q[2]))
    m2 = 4*q[2]
    m1 = mp.sqrt(14)*q[1]/3+2*q[2]
    m0 = q[0]+mp.sqrt(14)*q[1]/3+q[2]
    coefficients = [m0-3*m1/x+2*m2/(x*x), 4*m1/x-4*m2/(x*x), 2*m2/(x*x)-m1/x]
    parameters = [mp.mpf(0), x/2, x]
    projected = mp.matrix([[ (1-y/2)**2 for y in parameters ],
                           [3/(2*mp.sqrt(14))*(2*y-y*y) for y in parameters],
                           [y*y/4 for y in parameters]])
    reconstruction = projected*mp.matrix(coefficients)
    residual = mp.sqrt(sum(abs(reconstruction[j]-q[j])**2 for j in range(3)))
    error = mp.sqrt(sum(abs(q[j]-original_q[j])**2 for j in range(3)))
    lower = 4*max(mp.mpf(0), abs(original_q[2])-mp.mpf(float(error_allowance)))/(cap*cap)
    energies = [18*y*y+4*(coupling-9)*y for y in parameters]
    return {"t": float(t), "delta": float(delta), "error_allowance": float(error_allowance),
            "x_cap": float(cap), "construction_x": float(x),
            "lower_cost": float(lower), "upper_cost": float(sum(abs(c) for c in coefficients)),
            "coefficient_decimal_pairs": [[mp.nstr(c.real, digits), mp.nstr(c.imag, digits)] for c in coefficients],
            "x_decimal": [mp.nstr(y, digits) for y in parameters],
            "target_hopping_coordinates_decimal_pairs": [[mp.nstr(c.real, digits), mp.nstr(c.imag, digits)] for c in original_q],
            "high_precision_reconstruction_error": float(residual),
            "approximation_error_used": float(error), "maximum_component_energy_excess": float(max(energies)),
            "orbit_count": 3, "actual_gaussian_terms": "continuous Spin(7) orbits; no finite-branch-count claim",
            "uses_out_of_window_fock_repair": False,
            "arithmetic": f"explicit radical formulas at {digits} decimal digits; not directed interval arithmetic",
            "bound_scope": "complete eight-mode pure Gaussian dictionary in the stated energy window",
            "strict_interval_certified": False}


def restricted_support_upper(z, t, delta):
    """Analytic window envelope for a singlet witness; z need not be normalized.

The N=0,4,8 amplitudes obey 1,sqrt(3/2)*x,x^2/4. This uses the
second elementary symmetric polynomial of four Gaussian pair occupancies.
Floating evaluation only; no global numerical optimization is substituted.
"""
    coefficients = hopping_eigenbasis().T@np.asarray(z)
    x = x_cap(t, delta)
    return float(abs(coefficients[0])+math.sqrt(1.5)*abs(coefficients[1])*x+abs(coefficients[2])*x*x/4)

