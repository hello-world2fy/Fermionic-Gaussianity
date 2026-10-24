"""Sharp full-Gaussian support envelope for the two independent FK clusters."""
import numpy as np
from mpmath import iv
from .certificates import _bounds


def support_at_eta(eta):
    eta=np.asarray(eta)
    if np.any((eta<0)|(eta>1)):
        raise ValueError('eta must lie in [0,1]')
    return 1-eta/2


def necessity_certificate(cost_upper,eta_threshold,digits=70):
    if cost_upper<2 or not 0<eta_threshold<=1:
        raise ValueError('U>=2 and 0<eta0<=1 required')
    iv.dps=digits
    U,eta=iv.mpf(float(cost_upper)),iv.mpf(float(eta_threshold))
    fraction=2*(1-2/U)/eta
    absolute=2*(U-2)/eta
    mean_eta=2*(1-2/U)
    return {'U':float(cost_upper),'eta_threshold':float(eta_threshold),
            'restricted_support_upper':_bounds(1-eta/2)[1],
            'coefficient_mass_fraction_upper':min(1.,max(0.,_bounds(fraction)[1])),
            'absolute_coefficient_mass_upper':max(0.,_bounds(absolute)[1]),
            'coefficient_weighted_mean_eta_upper':max(0.,_bounds(mean_eta)[1]),
            'nontrivial':_bounds(fraction)[1]<1,
            'full_gaussian_dictionary':True,'interval_decimal_digits':digits,
            'witness':'2 M_A tensor M_B','L':2.,'global_support_exact':1.,
            'physical_envelope':'|<2 M_A M_B|g>| <= 1-eta(g)/2; sharp for every eta in [0,1]',
            'all_optimal_decompositions_have_product_components':True,
            'scope':'Exact t=0 target and exact Gaussian atoms; coefficient measure includes aligned phases'}
