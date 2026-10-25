"""Transfer-resolved coherences and count-preserving phase controls.

These are physical diagnostics in the fixed-pair sector, not canonical
distribution invariants or claims of a newly discovered mechanism.
"""
import numpy as np
from scipy.sparse import coo_matrix


def transfer_operator(model):
    """A = sum_{p,h} s_p^+ s_h^- / r, increasing m by one."""
    rows, cols = zip(*(model.transitions[p, h]
                      for p in range(model.r, model.L) for h in range(model.r)))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return coo_matrix((np.full(len(rows), 1/model.r), (rows, cols)),
                      shape=(model.dim, model.dim)).tocsr()


def resolved_observables(model, state, operator=None):
    operator = transfer_operator(model) if operator is None else operator
    density = abs(state)**2
    count = np.bincount(model.m, weights=density, minlength=model.r+1)
    local = state.conj()*(operator@state)
    # The action is indexed by destination sector m+1.
    coherence = (np.bincount(model.m, weights=local.real, minlength=model.r+1)
                 + 1j*np.bincount(model.m, weights=local.imag, minlength=model.r+1))[1:]
    current = -2*model.g*model.r*coherence.imag
    count_rate = np.r_[0., current]-np.r_[current, 0.]
    mean = float(count@np.arange(model.r+1))
    mean_rate = float(current.sum())
    variance_rate = float((2*np.arange(model.r)+1)@current-2*mean*mean_rate)
    # Independent Schrodinger-equation derivative, not the continuity formula.
    probability_rate = 2*np.real(state.conj()*(-1j*(model.H@state)))
    direct = np.bincount(model.m, weights=probability_rate, minlength=model.r+1)
    m = np.arange(model.r)
    bound = (m+1)*(model.r-m)/model.r*np.sqrt(count[:-1]*count[1:])
    return {'count': count.tolist(), 'mean': mean,
            'variance': float(count@np.arange(model.r+1)**2-mean**2),
            'coherence_real': coherence.real.tolist(), 'coherence_imag': coherence.imag.tolist(),
            'current': current.tolist(), 'mean_rate': mean_rate, 'variance_rate': variance_rate,
            'count_rate': count_rate.tolist(), 'current_absolute_sum': float(abs(current).sum()),
            'continuity_error': float(np.max(abs(direct-count_rate))),
            'coherence_bound_excess': float(np.max(abs(coherence)-bound))}


def sector_components(model, state):
    """Columns P_m|psi>, whose outer products give number-sector dephasing.

    No normalization: column squared norms are the exact mixing weights.
    """
    return state[:, None]*(model.m[:, None] == np.arange(model.r+1))


def phase_align(model, state, operator=None):
    """Keep every sector state and probability, align adjacent coherence phases.

    This deliberately changes the collective four-point function's magnitude;
    it is a diagnostic intervention, not a physical time evolution.
    """
    obs = resolved_observables(model, state, operator)
    coherence = np.array(obs['coherence_real'])+1j*np.array(obs['coherence_imag'])
    common = np.angle(coherence.sum())
    phase = np.r_[0., np.cumsum(np.angle(coherence)-common)]
    return state*np.exp(1j*phase[model.m])


def conditional_rank_one(model, state):
    """Best blockwise Schmidt direction, rescaled to preserve p(m) exactly."""
    answer = np.zeros_like(state)
    for _, indices in model.blocks:
        block = state[indices]
        u, s, vh = np.linalg.svd(block, full_matrices=False)
        answer[indices] = np.linalg.norm(s)*u[:, :1]*vh[:1, :]
    return answer
