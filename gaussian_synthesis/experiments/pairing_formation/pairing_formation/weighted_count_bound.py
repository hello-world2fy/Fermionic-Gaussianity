"""Fixed-number one-body fluctuation bound for arbitrary-orbital pure AGP.

For N=2M, A=c^dagger K c and Hermitian K:
Var(A) <= 2 Tr(gamma K^2) - Tr(gamma K)^2/M.
The proof uses concavity of e_M**(1/M), not an onsite restriction on K.
The functions below evaluate energy-diagonal observables on paired targets.
"""
import numpy as np
from .orbital_fluctuations import reduced_moments


def covariance_excess(occupations, covariance, occupied_pairs):
    n=np.asarray(occupations)
    return covariance-np.diag(n)+np.outer(n,n)/occupied_pairs


def weighted_statistics(model,state,weights):
    moments=reduced_moments(model,state);n=moments['occupations'];w=np.asarray(weights)
    mean=float(n@w);variance=float(w@moments['covariance']@w)
    bound=float(n@(w*w)-mean**2/model.r)
    return {'mean':mean,'variance':variance,'pure_PG_variance_bound':bound,'excess':variance-bound}


def full_number_spectrum(weights,occupied_pairs):
    """Exact spectra of X=sum w_i N_i/2 and Y=sum w_i^2 N_i/2.

    Every N_i=0,1,2 is retained, including broken pairs. Spin multiplicities
    do not change extrema. Exponential enumeration is only for small checks.
    """
    w=np.asarray(weights,dtype=float);L=len(w)
    if L>14:raise ValueError('This exact spectrum enumeration is limited to L<=14')
    numbers=np.arange(3**L,dtype=np.int64)
    counts=np.empty((len(numbers),L),dtype=np.int8)
    for j in range(L):
        counts[:,j]=numbers%3;numbers//=3
    counts=counts[counts.sum(axis=1)==2*occupied_pairs]
    return counts@w/2,counts@(w*w)/2


def weighted_fidelity_bound(model,state,weights):
    """Global pure-PG overlap bound; analytic inequality evaluated in float64."""
    result=weighted_statistics(model,state,weights);delta=result['excess']
    if delta<=0:
        return result|{'fidelity_upper_bound':1.,'infidelity_lower_bound':0.}
    X,Y=full_number_spectrum(weights,model.r);a=1-1/model.r
    operator=X*X-Y-2*a*result['mean']*X
    spectral_range=float(np.ptp(operator));count_range=float(np.ptp(X))
    distance=2*delta/(spectral_range+np.sqrt(spectral_range**2+4*a*count_range**2*delta))
    return result|{'fidelity_upper_bound':float(1-distance**2),
                   'infidelity_lower_bound':float(distance**2),
                   'trace_distance_lower_bound':float(distance),
                   'operator_spectral_range':spectral_range,'count_spectral_range':count_range,
                   'scope':'All normalized total-number-projected pure Gaussians; no extra many-body projection; float64, not interval arithmetic.'}


def weighted_echo(model,state,weights,angles):
    state=np.asarray(state)/np.linalg.norm(state);p=abs(state)**2
    X=model.occ@np.asarray(weights);angles=np.asarray(angles)
    echo=abs(np.exp(-1j*angles[:,None]*X)@p)**2
    bound=weighted_statistics(model,state,weights)['pure_PG_variance_bound']
    return {'angles':angles,'overlap':echo,'excess':1-echo-angles**2*bound}
