"""Separate radial freedom from conditional interband entanglement.

A single pair direction with arbitrary radial weights is contained in the
class of states of Schmidt rank one in every fixed-m block. The block-SVD
fidelity upper bound is therefore independent of all radial coefficient fits.
It is not a bound for unrestricted number-projected fermionic Gaussians.
"""
import numpy as np


def collective_basis(model, particle, hole):
    columns = np.zeros((model.dim, model.r+1), complex)
    for m, (subsets, indices) in enumerate(model.blocks):
        p = np.array([np.prod(particle[list(c)]) for c in subsets])
        h = np.array([np.prod(hole[list(c)]) for c in subsets])
        block = p[:, None]*h[None, :]
        norm = np.linalg.norm(block)
        if norm:
            columns[indices, m] = block/norm
    return columns


def analyze(model, state):
    spectra, best = [], 0.
    for _, indices in model.blocks:
        s = np.linalg.svd(state[indices], compute_uv=False)
        spectra.append((s*s).tolist())
        best += s[0]**2
    _, indices = model.blocks[1]
    u, s, vh = np.linalg.svd(state[indices], full_matrices=False)
    particle, hole = u[:, 0], vh[0]
    columns = collective_basis(model, particle, hole)
    projected = columns@(columns.conj().T@state)
    weight = float(np.vdot(projected, projected).real)
    result = {'conditional_rank1_fidelity_upper': float(best),
              'single_direction_radial_fidelity': weight,
              'conditional_schmidt_weights': spectra,
              'm1_rank1_weight': float(s[0]**2/sum(s*s)) if sum(s*s) else 1.,
              'm1_particle_neff': float(1/sum(abs(particle)**4)),
              'm1_hole_neff': float(1/sum(abs(hole)**4))}
    return result, projected/np.sqrt(weight), columns


def single_pbcs_count_distance_lower(count):
    """A conservative trace-distance lower bound from Var(m)<=mean(m).

For m in [0,r], |mean_p-mean_q|<=r TV and
|<m(m-1)>_p-<m(m-1)>_q|<=r(r-1) TV. Thus positive
Var_p(m)-mean_p <= (3r^2-r) TV for any onsite PBCS q.
Data processing TV<=pure-state trace distance then bounds fidelity.
"""
    p = np.asarray(count)
    r = len(p)-1
    m = np.arange(r+1)
    mean = p@m
    excess = p@(m*(m-1))-mean**2
    distance = max(0., float(excess))/(3*r*r-r)
    return {'trace_distance_lower': distance, 'fidelity_upper': 1-distance**2}
