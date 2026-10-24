"""Exact finite Gaussian projector for the t=0 two-magic target.

The local target is (|7>-|8>)/sqrt(2), including its odd parity.
Sixteen signed even Majorana monomials stabilize it and sum to its
rank-one projector. Their 256 tensor products are actual even Gaussian
unitaries; they are not single Gaussian atoms after averaging.
"""
from functools import lru_cache
import numpy as np
from . import paths
from fk.model import build_model


MASKS=np.array([0,15,51,60,85,90,102,105,150,153,165,170,195,204,240,255])
SIGNS=np.array([1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1])


@lru_cache(None)
def proof_and_actions():
    model=build_model()
    local=[]
    for mask,sign in zip(MASKS,SIGNS):
        U=np.eye(16,dtype=complex)
        for a in range(8):
            if int(mask)>>a&1:U=U@model.local_gammas[a]
        U*=sign
        if np.max(abs(U.real-np.rint(U.real))) or np.max(abs(U.imag-np.rint(U.imag))):
            raise ArithmeticError('Majorana monomials must be Gaussian integers')
        local.append(U)
    local=np.array(local)
    target_integer=np.zeros(16,dtype=int);target_integer[7]=1;target_integer[8]=-1
    # Sum U = 16 |M><M| = 8 v v^T, checked as exact Gaussian integers.
    expected=8*np.outer(target_integer,target_integer)
    assert np.array_equal(local.sum(axis=0),expected)
    assert np.array_equal(local@target_integer,np.tile(target_integer,(16,1)))
    assert np.array_equal(local@local.conj().transpose(0,2,1),np.tile(np.eye(16),(16,1,1)))
    global_actions=np.array([np.kron(B,A)[np.ix_(model.even_masks,model.even_masks)] for B in local for A in local])
    v=np.kron(target_integer,target_integer)[model.even_masks]
    assert np.array_equal(global_actions.sum(axis=0),64*np.outer(v,v))
    permutations=np.argmax(abs(global_actions),axis=2)
    phases=np.take_along_axis(global_actions,permutations[:,:,None],axis=2)[:,:,0]
    return {'exact_Gaussian_integer_identity_checked':True,'local_masks':MASKS.tolist(),
            'local_signs':SIGNS.tolist(),'local_parity':'odd','local_unitary_count':16,
            'global_actual_Gaussian_branches_per_particle':256,
            'target_exact': '(e7-e8) tensor (e7-e8) / 2; A is low-bit factor',
            'projector_identity': 'sum_ab U_B,b tensor U_A,a = 256 |s0><s0|',
            'Gaussianity_proof':'Every U is a signed product of an even number of Majoranas, hence an even Gaussian unitary, up to ray phase.'},permutations,phases,v/2


def orbit(ket):
    _,permutations,phases,_=proof_and_actions()
    return phases*np.asarray(ket)[permutations]


def aligned_orbit_decomposition(kets):
    """Factored explicit 256N-term decomposition, exact before rounding.

For a=<g|s0>, P(g a/|a|)=|a| s0. Thus averaging the actual
Gaussian branches and dividing by mean |a| exactly reconstructs s0.
The cost is 1/mean|a|; no projected ket is counted as a Gaussian atom.
"""
    proof,_,_,s0=proof_and_actions()
    overlaps=np.asarray(kets).conj()@s0
    if np.any(abs(overlaps)==0):raise ValueError('zero overlap has no alignment phase')
    mean=float(np.mean(abs(overlaps)))
    cost=1/mean
    factors=cost*overlaps/abs(overlaps)/(len(kets)*256)
    return factors,{'coefficient_cost':cost,'actual_Gaussian_terms':int(256*len(kets)),
                    'orbit_branches_per_particle':256,'mean_absolute_overlap':mean,
                    'exact_algebraic_reconstruction':True,
                    'canonical_measure_approximation':'SMC P particles; finite realized decomposition is exact by its independent projector identity',
                    'floating_cost_not_interval_certified':True,'proof':proof}
