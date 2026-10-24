"""Physical stability consequences of the joint witness envelope.

This module initially supplies the exact scalar envelope; contact search,
symmetry quotient classification and high-precision Hessians remain separate
research requirements, not inferred from this one-dimensional stability law.
"""
import numpy as np


def joint_envelope(sqrt_eta,lam):
    x=np.asarray(sqrt_eta)
    if lam<4 or np.any((x<0)|(x>1)):raise ValueError('lambda>=4, sqrt_eta in[0,1]')
    return 1+lam+2*x+(1-lam/2)*x*x


def completed_square(sqrt_eta,lam):
    return 1+lam+2/(lam-2)-(lam-2)/2*(np.asarray(sqrt_eta)-2/(lam-2))**2
