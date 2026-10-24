"""Actual Gaussian dictionaries, coherent kernels and target-free prediction.

No target wavefunction is accepted by the variational solver or candidate
selection. Exact diagonalization belongs only in the external evaluator.
"""
import time
import numpy as np
from scipy.linalg import qr
from scipy.sparse import csr_matrix
from .tempering import actions
from . import paths
from fk.observables import exact_metrics, mass_correlation_wick
from fk3.discovery import covariance_from_kets


PERTURBATION=np.array([1,1,1,1,-1,-1,-1,-1])


def perturbed_hamiltonian(model,t,epsilon,*,even_only=True):
    hopping=np.zeros((256,256),complex)
    for j,r in enumerate(PERTURBATION):
        hopping+=.5j*float(t)*(1+float(epsilon)*r)*(model.gammas[j]@model.gammas[j+8])
    full=model.h_int+hopping
    return full[np.ix_(model.even_masks,model.even_masks)] if even_only else full


def diverse_order(kets,maximum):
    """Pivot actual ket columns; no singlet projection or target-assisted fit."""
    states=np.asarray(kets,complex)
    _,_,pivots=qr(states.T,pivoting=True,mode='economic')
    return pivots[:min(int(maximum),len(states))]


def coherent_kernels(kets,operators):
    phi=np.asarray(kets,complex).T
    result={'overlap':phi.conj().T@phi}
    result.update({name:phi.conj().T@operator@phi for name,operator in operators.items()})
    return result


def predict(kets,hamiltonian,*,rank_tolerance=1e-10):
    """Generalized eigenproblem in a frozen dictionary, independent of ED."""
    started=time.monotonic()
    phi=np.asarray(kets,complex).T
    S=phi.conj().T@phi
    Hp=phi.conj().T@hamiltonian@phi
    eigenvalues,V=np.linalg.eigh(.5*(S+S.conj().T))
    keep=eigenvalues>rank_tolerance*max(1.,float(eigenvalues[-1]))
    if not np.any(keep):
        raise ArithmeticError('dictionary has no retained overlap direction')
    transform=V[:,keep]/np.sqrt(eigenvalues[keep])[None,:]
    reduced=transform.conj().T@Hp@transform
    E,vec=np.linalg.eigh(.5*(reduced+reduced.conj().T))
    coefficients=transform@vec[:,0]
    state=phi@coefficients
    norm=np.linalg.norm(state)
    state/=norm
    coefficients/=norm
    residual=Hp@coefficients-float(E[0])*(S@coefficients)
    return {'energy':float(np.vdot(state,hamiltonian@state).real),
            'actual_gaussian_terms':len(kets),'retained_rank':int(keep.sum()),
            'rank_tolerance':rank_tolerance,'smallest_retained_overlap_eigenvalue':float(eigenvalues[keep][0]),
            'generalized_eigen_residual':float(np.linalg.norm(residual)),
            'elapsed_seconds':time.monotonic()-started,
            'uses_test_target':False}, {'state':state,'coefficients':coefficients,
                                       'overlap_kernel':S,'hamiltonian_kernel':Hp}


def coherent_and_incoherent_metrics(kets,coefficients,hamiltonian,model,*,kernels=None,covariances=None):
    """Compare a coherent state and the mixture with weights proportional |c|²."""
    c=np.asarray(coefficients)
    state=c@kets
    state/=np.linalg.norm(state)
    weights=abs(c)**2
    weights/=weights.sum()
    masks=model.even_masks
    mass=model.mass_correlation_operator[np.ix_(masks,masks)]
    kernels=coherent_kernels(kets,{'hamiltonian':hamiltonian,'mass_correlation':mass}) if kernels is None else kernels
    cov=covariance_from_kets(kets) if covariances is None else covariances
    mixture_cov=np.einsum('i,ijk->jk',weights,cov)
    mixture_mass=float(weights@np.diag(kernels['mass_correlation']).real)
    mixture_energy=float(weights@np.diag(kernels['hamiltonian']).real)
    metrics=exact_metrics(state,model)
    metrics['energy']=float(np.vdot(state,hamiltonian@state).real)
    denominator=float(np.vdot(c,kernels['overlap']@c).real)
    coherent_parts={}
    for name,K in kernels.items():
        diagonal=float(np.sum(abs(c)**2*np.diag(K)).real/denominator)
        total=float(np.vdot(c,K@c).real/denominator)
        coherent_parts[name]={'diagonal':diagonal,'off_diagonal':total-diagonal,'total':total}
    return {'coherent':metrics,
            'incoherent':{'energy':mixture_energy,'mass_correlation':mixture_mass,
                          'wick_prediction':float(mass_correlation_wick(mixture_cov,model.j_matrices)),
                          'wick_deviation':mixture_mass-float(mass_correlation_wick(mixture_cov,model.j_matrices)),
                          'weight_definition':'|c_i|²/sum |c_i|²; no overlap cross terms'},
            'coherent_kernel_parts':coherent_parts},kernels


def optimize_gaussian_energy(hamiltonian,starts,*,sweeps=80,tolerance=1e-8):
    """Multistart full-manifold coordinate descent; a local HFB baseline."""
    started=time.monotonic()
    H=csr_matrix(hamiltonian)
    states=np.asarray(starts,complex).copy()
    Hg=(H@states.T).T
    permutations,phases=actions(8)
    histories=[]
    converged=np.zeros(len(states),bool)
    for sweep in range(sweeps):
        for permutation,phase in zip(permutations,phases):
            transformed=phase*states[:,permutation]
            Htransformed=(H@transformed.T).T
            a=np.einsum('ij,ij->i',states.conj(),Hg).real
            b=np.einsum('ij,ij->i',transformed.conj(),Htransformed).real
            cross=np.einsum('ij,ij->i',states.conj(),Htransformed).real
            theta=.5*np.arctan2(-2*cross,b-a)
            theta[converged]=0
            c,s=np.cos(theta)[:,None],np.sin(theta)[:,None]
            states=c*states+s*transformed
            Hg=c*Hg+s*Htransformed
        energies=np.einsum('ij,ij->i',states.conj(),Hg).real
        histories.append(energies.copy())
        gradients=[]
        for permutation,phase in zip(permutations,phases):
            gradients.append(2*np.einsum('ij,ij->i',(phase*states[:,permutation]).conj(),Hg).real)
        gradient=np.max(abs(np.array(gradients)),axis=0)
        converged=gradient<tolerance
        if np.all(converged):
            break
    states/=np.linalg.norm(states,axis=1)[:,None]
    energies=np.einsum('ij,ij->i',states.conj(),(H@states.T).T).real
    return states,{'energies':energies,'energy_history':np.array(histories),
                   'gradient_max':gradient,'converged':converged,'elapsed_seconds':time.monotonic()-started,
                   'sweeps':sweep+1,'globally_optimal_proved':False,'actual_terms_per_solution':1}
