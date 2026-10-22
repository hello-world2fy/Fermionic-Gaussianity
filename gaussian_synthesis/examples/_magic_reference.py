"""Analytic benchmark only. Never imported by the optimization package."""

import math
import numpy as np


def magic_state(basis):
    basis.require_four_mode_even()
    return (basis.ket(0)+basis.ket(15))/np.sqrt(2)


def magic_moment(s):
    return 2.**(-s)*6/((s+1)*(s+2)*(s+3))


def magic_q(k):
    return 2.**(-k)*(k+3)*(k+4)*(k+5)/60


def magic_optimizer(k, basis):
    return magic_q(k)**(-1/(2*k-1))*magic_state(basis)


def magic_mass(k):
    return math.sqrt(2)*(k+1)*(k+2)*(k+3)/((k+.5)*(k+1.5)*(k+2.5))
