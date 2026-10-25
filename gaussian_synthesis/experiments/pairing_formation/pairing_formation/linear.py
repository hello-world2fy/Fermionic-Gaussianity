"""Finite-size linear spin response and its bosonic squeezed-vacuum lift.

Diagonal self-interaction is retained before linearization. Replacing s_i^z
by its Fermi-sea value in the i=j term would give an incorrect frequency.
"""
import numpy as np
from scipy.integrate import solve_ivp


def matrices(model):
    r, g, eps = model.r, model.g, model.energies
    particle = np.diag(2*eps[r:])-g*np.ones((r, r))
    hole = np.diag(-2*eps[:r]+2*g)-g*np.ones((r, r))
    source = -g*np.ones((r, r))
    response = np.block([[particle, source], [-source.conj().T, -hole.T]])
    return particle, hole, source, response


def unstable_modes(model):
    _, _, _, response = matrices(model)
    values, vectors = np.linalg.eig(response)
    order = np.argsort(values.imag)[::-1]
    output = []
    for i in order:
        if values[i].imag <= 1e-9:
            continue
        p, h = vectors[:model.r, i], vectors[model.r:, i].conj()
        p, h = p/np.linalg.norm(p), h/np.linalg.norm(h)
        output.append({'frequency_real': float(values[i].real),
                       'growth_rate': float(values[i].imag),
                       'particle_neff': float(1/sum(abs(p)**4)),
                       'hole_neff': float(1/sum(abs(h)**4))})
    return output


def squeezed_pair_matrix(model, times):
    particle, hole, source, _ = matrices(model)
    r = model.r
    def rhs(t, flat):
        T = flat.reshape(r, r)
        return (-1j*(source+particle@T+T@hole.T+T@source.conj().T@T)).ravel()
    if times[-1] == 0:
        return np.zeros((len(times), r, r), complex)
    sol = solve_ivp(rhs, (0, times[-1]), np.zeros(r*r, complex), t_eval=times,
                    method='DOP853', rtol=2e-11, atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.reshape(-1, r, r)


def permanent(matrix):
    """Subset dynamic program, exact algebra up to floating point (small r)."""
    n = len(matrix)
    values = np.zeros(1 << n, complex)
    values[0] = 1
    for mask in range(1, 1 << n):
        row = mask.bit_count()-1
        values[mask] = sum(matrix[row, j]*values[mask ^ (1 << j)]
                           for j in range(n) if mask & (1 << j))
    return values[-1]


def coupled_cluster_state(model, T):
    """exp(sum T_ph s_p^+ s_h^-) |F>; amplitudes are permanents."""
    state = np.empty(model.dim, complex)
    state[model.fermi_index] = 1.
    for current, particle, holes, parents in model.permanent_recursion:
        state[current] = np.sum(T[particle[:, None], holes]*state[parents], axis=1)
    return state/np.linalg.norm(state)


def rank_truncate(T, q):
    u, s, vh = np.linalg.svd(T, full_matrices=False)
    return (u[:, :q]*s[:q])@vh[:q], s
