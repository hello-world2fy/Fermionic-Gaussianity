"""Exact rational coercivity for k2 on fixed-number paired amplitudes.

These constants apply to the compressed Cartan map on the declared paired
Hilbert space, not to every state of the complete physical Fock space.
"""
from fractions import Fraction
from math import factorial,comb
import numpy as np


def paired_cartan_floor(levels):
    rows=[]
    for s in range(0,levels+1,2):
        def kernel(distance):
            return Fraction(factorial(2*distance)*factorial(2*(s-distance)),
                4**s*factorial(distance)*factorial(s-distance)*factorial(s))
        row_sum=sum(Fraction(comb(s//2,j)**2)*kernel(2*j) for j in range(s//2+1))
        floor=Fraction(1) if s==0 else 4*kernel(0)-row_sum
        if floor<=0:raise ValueError('This Gershgorin argument gives no positive bound at this size')
        rows.append({'different_levels':s,'floor':floor,'row_sum':row_sum})
    return min(row['floor'] for row in rows),rows


def numerical_image_obstruction(levels,witness,basis,residual):
    """Evaluate an exact analytic inequality using floating input diagnostics.

    This routine is NOT interval arithmetic and does not certify the exact
    physical evolution. It reports how much saved numeric inputs separate a
    target from C2(basis), allowing the supplied synthesis residual.
    """
    mu,rows=paired_cartan_floor(levels)
    fitted=np.linalg.lstsq(basis,witness,rcond=None)[0]
    distance=float(np.linalg.norm(witness-basis@fitted))
    separation=max(0.,float(mu)*np.linalg.norm(witness)**2*distance/8-residual)
    ceiling=max(0.,1-separation**2/2)**2
    return {'exact_rational_mu':str(mu),'mu':float(mu),'witness_subspace_distance':distance,
        'synthesis_residual':float(residual),'phase_aligned_distance_lower_expression':float(separation),
        'fidelity_upper_expression':float(ceiling),'infidelity_lower_expression':float(1-ceiling),
        'evaluation_is_interval_certificate':False,
        'block_floors':[{'s':r['different_levels'],'floor':str(r['floor'])} for r in rows]}
