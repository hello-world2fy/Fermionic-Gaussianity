"""Two distinct mean-field conventions and explicit number projection."""

import numpy as np
from scipy.integrate import solve_ivp
from gaussian_synthesis.metrics import normalized
from .exact import checked_times


def initial_spinors(basis):
    p = basis.m/basis.L
    return np.tile([np.sqrt(1-p), np.sqrt(p)], (basis.L, 1)).astype(complex)


def project_spinors(basis, spinors):
    spinors = np.asarray(spinors, dtype=complex)
    if spinors.shape != (basis.L, 2) or not np.all(np.isfinite(spinors)):
        raise ValueError('Expected finite spinors with shape (L,2)')
    factors = np.where(basis.occupations, spinors[:, 1], spinors[:, 0])
    return normalized(np.prod(factors, axis=1))


def spinor_hamiltonians(energies, spinors, g, convention):
    if convention not in ('td_bcs', 'product_tdvp'):
        raise ValueError('Unknown mean-field convention')
    kappa = spinors[:, 0].conj()*spinors[:, 1]
    delta = np.full(len(energies), g*kappa.sum(), dtype=complex)
    upper = np.asarray(energies, dtype=float).copy()
    if convention == 'product_tdvp':
        delta -= g*kappa
        upper -= g
    matrices = np.zeros((len(energies), 2, 2), dtype=complex)
    matrices[:, 0, 0], matrices[:, 1, 1] = -energies, upper
    matrices[:, 0, 1], matrices[:, 1, 0] = -delta.conj(), -delta
    return matrices


def mean_field_energy(energies, spinors, g, convention):
    kappa = spinors[:, 0].conj()*spinors[:, 1]
    p = np.abs(spinors[:, 1])**2
    if convention == 'td_bcs':
        interaction = abs(kappa.sum())**2
    elif convention == 'product_tdvp':
        interaction = abs(kappa.sum())**2+np.sum(p**2)
    else:
        raise ValueError('Unknown mean-field convention')
    return float(2*np.dot(energies, p-.5)-g*interaction)


def evolve_mean_field(model, g, times, *, convention='td_bcs', rtol=1e-11, atol=1e-13):
    times = checked_times(times)
    initial = initial_spinors(model.basis)
    # Validate the convention even for the zero-time shortcut.
    spinor_hamiltonians(model.energies, initial, g, convention)
    if times[-1] == 0:
        return initial[None, :, :]
    def rhs(t, flat):
        spinors = flat.reshape(model.basis.L, 2)
        h = spinor_hamiltonians(model.energies, spinors, g, convention)
        return (-1j*np.einsum('bij,bj->bi', h, spinors)).ravel()
    solution = solve_ivp(rhs, (0., times[-1]), initial.ravel(), method='DOP853',
                         t_eval=times, rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T.reshape(len(times), model.basis.L, 2)
