"""Coherent moments of actual full-Gaussian single-mode occupations.

For p_i(g)=<g|n_i|g> and k=2, the insertion in the Cartan projector has
three eigenvalues (n-1+j)_d/(2n)_d, j=0,1,2. Here n counts physical
complex fermionic modes. This is not an occupation in a projected atom,
nor the physical expectation in the reconstructed target. See
OCCUPATION_SYMBOL_RESPONSE.md for the branching/Beta-integral derivation.
"""
import numpy as np
from .coherent_partition import coherent_parts


def beta_coefficients(physical_modes, degree):
    if physical_modes < 2 or degree < 0 or int(degree) != degree:
        raise ValueError('Need n>=2 and a nonnegative integer degree')
    result = np.ones(3)
    for d in range(int(degree)):
        result *= (physical_modes-1+np.arange(3)+d)/(2*physical_modes+d)
    return result


def occupation_components(synthesis, occupation, anomalous):
    """Contract the j=0,1,2 copy-occupation sectors.

    occupation is the diagonal of physical n_i, with entries 0 or 1;
    anomalous is B_u(n_i u). All vectors use the same output sector.
    """
    occupation = np.asarray(occupation)
    if not np.all((occupation == 0) | (occupation == 1)):
        raise ValueError('The mode occupation must be a projector, not an arbitrary density')
    both = occupation*anomalous
    one = occupation*synthesis+anomalous-2*both
    zero = synthesis-one-both
    return np.stack((zero, one, both))


def occupation_moments(replica, witness, occupation, degrees=(0,1,2)):
    """Exact Cartan contractions in floating point; no Monte Carlo."""
    _, C = replica.evaluate(witness)
    parts = coherent_parts(replica, witness, occupation*witness)
    components = occupation_components(C, occupation, parts['conjugate_ratio'])
    n = 2*replica.model.L
    moments = np.array([beta_coefficients(n,d)@components for d in degrees])
    ward = 2*parts['complex_ratio']-parts['conjugate_ratio']-occupation*C
    return moments, {'ward_residual':float(np.linalg.norm(ward))}, components


def orthogonal_span(columns, cutoff=1e-11):
    columns = np.asarray(columns,complex)
    if columns.ndim == 1: columns=columns[:,None]
    norms = np.linalg.norm(columns,axis=0)
    columns = columns[:,norms>1e-14]/norms[norms>1e-14]
    if columns.shape[1] == 0: return np.empty((len(columns),0),complex)
    U,s,_=np.linalg.svd(columns,full_matrices=False)
    return U[:,s>cutoff*s[0]]
