"""Candidate exchange-assisted two-transfer mode of the flat two-source model.

The formula was inferred from r=7,9 fifth-order coefficients. Validation is
recorded separately; this module is not a general-size theorem or a Gaussian
distribution discovery. b is bright and d is its orthogonal third-harmonic
pair partner. The two-band combination has Schmidt rank two.
"""
import numpy as np
from itertools import combinations


def return_pair_mode(r):
    if r < 5:
        raise ValueError('The return-mode formula is defined here only for r>=5')
    subsets=np.array(list(combinations(range(r),2)))
    theta=2*np.pi*np.arange(r)/r
    difference=theta[subsets[:,0]]-theta[subsets[:,1]]
    bright=np.cos(difference)
    dark=np.cos(3*difference)+2/(r-2)*bright
    mode=np.outer(dark,bright)+np.outer(bright,dark)
    mode/=np.linalg.norm(mode)
    return bright,dark,mode


def fifth_order_coefficient(r,G=4.):
    bright,dark,_=return_pair_mode(r)
    return 1j*G**5*(r-3)/(15*r**5)*(np.outer(dark,bright)+np.outer(bright,dark))
