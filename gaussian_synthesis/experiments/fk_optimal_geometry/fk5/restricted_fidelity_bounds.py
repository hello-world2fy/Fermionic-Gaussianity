"""Directed cost-limited energy-window fidelity and energy obstructions."""
from fractions import Fraction
from mpmath import iv
from . import paths
from fk4.certificates import _bounds


def restricted_bound(t,delta,budget,L0_lower,gap_lower):
    if t<9 or budget<1:raise ValueError('This window theorem assumes t>=9 and B>=1')
    iv.dps=70
    delta=Fraction(delta);D=iv.mpf(delta.numerator)/delta.denominator
    a=4*(iv.mpf(float(t))-9)
    X=2*D/(iv.sqrt(a*a+72*D)+a)
    if _bounds(X)[1]>=1:raise ValueError('Endpoint support formula used only for X<1')
    h=1-(1-X)**2/2
    deficit=iv.mpf(float(L0_lower))-iv.mpf(float(budget))*h
    if _bounds(deficit)[0]<=0:d=iv.mpf(0)
    else:d=deficit/2
    dlow=max(0.,_bounds(d)[0])
    if dlow>2**.5:raise ArithmeticError('The admissible normalized class is empty')
    dd=iv.mpf(dlow)
    fidelity=min(1.,_bounds((1-dd*dd/2)**2)[1])
    energy=max(0.,_bounds(iv.mpf(float(gap_lower))*(dd*dd-dd**4/4))[0])
    return {'t':float(t),'delta_numerator':delta.numerator,'delta_denominator':delta.denominator,
        'delta':float(delta),'coefficient_budget':float(budget),'window_X_interval':_bounds(X),
        'restricted_support_upper':_bounds(h)[1],'physical_overlap_lower':float(L0_lower),
        'phase_aligned_state_distance_lower':dlow,'fidelity_upper':fidelity,
        'Hamiltonian_gap_lower':float(gap_lower),'ground_energy_error_lower':energy,
        'nontrivial':dlow>0,'arbitrary_number_of_Gaussian_terms':True,
        'normalized_approximate_state_required':True,
        'scope':'All normalized states with every actual Gaussian inside the energy window and coefficient l1 cost<=budget'}
