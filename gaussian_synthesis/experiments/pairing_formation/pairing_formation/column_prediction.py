"""Predictions and coherent-kernel ablations in frozen symmetry-orbit spans."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


def orthonormal_span(atoms, tolerance=1e-10):
    u, s, vh = np.linalg.svd(atoms, full_matrices=False)
    return u[:, s > tolerance*s[0]], s


def pool_pursuit(atoms, target, count):
    chosen, records = [], []
    residual = target.copy()
    for k in range(count):
        score = abs(atoms.conj().T@residual)
        score[chosen] = -1
        j = int(np.argmax(score))
        chosen.append(j)
        A = atoms[:, chosen]
        coef = np.linalg.lstsq(A, target, rcond=1e-11)[0]
        residual = target-A@coef
        records.append({'columns': k+1, 'fidelity': float(1-np.linalg.norm(residual)**2),
                        'mass': float(np.sum(abs(coef)))})
    return np.array(chosen), records


def evolve_span(model, Q, initial_coefficients, elapsed, transport='static'):
    """Common one-body transport is fixed by H, not fitted to a future state."""
    if transport == 'static':
        energy, vectors = np.linalg.eigh(Q.conj().T@(model.H@Q))
        weights = vectors.conj().T@initial_coefficients
        states = (np.exp(-1j*np.asarray(elapsed)[:, None]*energy)*weights)@vectors.T@Q.T
        return states
    if transport != 'free':
        raise ValueError(transport)
    free_energy = 2*model.occ@model.energies-model.energies.sum()
    interaction = model.H-diags(free_energy)
    def rhs(t, c):
        phase = np.exp(-1j*free_energy*t)
        return -1j*(Q.conj().T@(phase.conj()*(interaction@(phase*(Q@c)))))
    if elapsed[-1] == 0:
        return (Q@initial_coefficients)[None]
    solution = solve_ivp(rhs, (0., elapsed[-1]), initial_coefficients, t_eval=elapsed,
                         method='DOP853', rtol=2e-10, atol=2e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    return (solution.y.T@Q.T)*np.exp(-1j*np.asarray(elapsed)[:, None]*free_energy)


def coherent_kernels(model, atoms, coefficients):
    """All off-diagonal terms retained; compare with deleting atom coherences."""
    state = atoms@coefficients
    norm = float(np.vdot(state, state).real)
    c = coefficients/np.sqrt(norm)
    S = atoms.conj().T@atoms
    O = atoms.conj().T@(model.m[:, None]*atoms)
    O2 = atoms.conj().T@((model.m**2)[:, None]*atoms)
    cross_action = np.zeros_like(atoms)
    for p in range(model.r, model.L):
        for h in range(model.r):
            dst, src = model.transitions[p, h]
            cross_action[dst] += atoms[src]/model.r
    K = atoms.conj().T@cross_action
    coherent_norm = float(np.vdot(c, S@c).real)
    diagonal_norm = float(np.sum(abs(c)**2*np.diag(S).real))
    def get(matrix):
        return np.vdot(c, matrix@c), np.sum(abs(c)**2*np.diag(matrix))/diagonal_norm
    mean, mixed_mean = get(O)
    second, mixed_second = get(O2)
    cross, mixed_cross = get(K)
    # Count distributions of the coherent state and of normalized diagonal mixture.
    coherent_count = model.observe(state/np.sqrt(norm))['count']
    mixed_count = np.array([np.sum(abs(c)**2*np.sum(abs(atoms[model.m == m])**2, axis=0))/diagonal_norm
                            for m in range(model.r+1)])
    return {'coherent_norm': coherent_norm, 'incoherent_normalization_before_rescaling': diagonal_norm,
            'coherent_mean': float(mean.real), 'incoherent_mean': float(mixed_mean.real),
            'coherent_variance': float(second.real-mean.real**2),
            'incoherent_variance': float(mixed_second.real-mixed_mean.real**2),
            'coherent_cross': [float(cross.real), float(cross.imag)],
            'incoherent_cross': [float(mixed_cross.real), float(mixed_cross.imag)],
            'count_TV_after_atom_dephasing': float(.5*np.sum(abs(np.array(coherent_count)-mixed_count))),
            'ablation_scope': 'delete coherences between nonorthogonal symmetry-orbit atoms and renormalize; initial count distribution generally changes'}
