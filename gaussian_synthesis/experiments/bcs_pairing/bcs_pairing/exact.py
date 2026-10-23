"""Small-sector exact references, kept separate from variational baselines."""

import numpy as np
from scipy.integrate import solve_ivp
from gaussian_synthesis.metrics import normalized


def checked_times(times, start=0.):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError('times must be a nonempty finite one-dimensional array')
    if np.any(times < start) or np.any(np.diff(times) <= 0):
        raise ValueError('times must increase strictly and not precede the initial time')
    return times


def evolve_constant(model, initial, g, times):
    times = checked_times(times)
    initial = normalized(model.basis.sector.vector(initial, nonzero=True))
    eigenvalues, eigenvectors = np.linalg.eigh(model.hamiltonian(g))
    weights = eigenvectors.conj().T @ initial
    return (np.exp(-1j*times[:, None]*eigenvalues)*weights) @ eigenvectors.T


def evolve_inverse_time(model, eta, t0, times, *, rtol=1e-10, atol=1e-12):
    """Launch the Dicke state at t0; remove only its known singular global phase.

    Integrate in log time with K replaced by K-m(L-m+1)I. The returned ket
    differs from the physical ket by exp[-i eta/L m(L-m+1) log(t/t0)].
    """
    if not np.isfinite(t0) or t0 <= 0 or not np.isfinite(eta) or eta <= 0:
        raise ValueError('t0 and eta must be finite and positive')
    times = checked_times(times, t0)
    initial = model.dicke()
    if times[-1] == t0:
        return initial[None, :]
    eigenvalue = model.basis.m*(model.basis.L-model.basis.m+1)
    shifted = model.K-eigenvalue*np.eye(model.basis.dimension)
    def rhs(tau, psi):
        return -1j*(t0*np.exp(tau)*(model.H0 @ psi)-eta/model.basis.L*(shifted @ psi))
    tau = np.log(times/t0)
    solution = solve_ivp(rhs, (0., tau[-1]), initial, method='DOP853', t_eval=tau, rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T
