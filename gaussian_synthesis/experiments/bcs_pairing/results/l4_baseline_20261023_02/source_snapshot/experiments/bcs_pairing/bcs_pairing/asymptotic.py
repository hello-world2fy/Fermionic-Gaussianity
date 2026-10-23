"""Zabalo et al., arXiv:2208.07320v2, Eqs. (42)-(43), and phase controls."""

import numpy as np
from gaussian_synthesis.metrics import normalized


def asymptotic_state(model, eta, *, alpha=1., time=0., frame='stripped', energy_reference=1.):
    if not np.isfinite(eta) or eta <= 0 or not np.isfinite(alpha) or not np.isfinite(time):
        raise ValueError('Require eta>0 and finite alpha/time')
    if frame not in ('stripped', 'physical') or not np.isfinite(energy_reference) or energy_reference <= 0:
        raise ValueError('Invalid frame or energy reference')
    eps, basis = model.energies, model.basis
    if np.any(np.diff(eps) <= 0):
        raise ValueError('The asymptotic formula requires strictly increasing nondegenerate energies')
    inverse_nu = eta/basis.L
    differences = np.abs(eps[:, None]-eps[None, :])/energy_reference
    np.fill_diagonal(differences, 1.)
    logs = np.log(differences)
    occupied = basis.occupations
    log_modulus = -np.pi*inverse_nu*(occupied @ np.arange(1, basis.L+1))
    phase = alpha*inverse_nu*np.einsum('bi,ij,bj->b', occupied, logs, occupied)
    if frame == 'physical':
        phase -= 2*time*(occupied @ eps)+inverse_nu*(occupied @ logs.sum(axis=1))
    return normalized(np.exp(log_modulus-log_modulus.max()+1j*phase))
