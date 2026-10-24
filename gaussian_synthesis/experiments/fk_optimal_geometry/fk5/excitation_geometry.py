"""Exact hopping-vacuum Fock basis and the Cayley four-form pairing bound."""
from itertools import combinations
import numpy as np
from .two_orbit import scaled_basis
from .joint_dual_certificates import exact_schmidt_polar


def exact_cayley_pairing_proof(model):
    U=np.zeros((256,256),complex);U[:,0]=exact_schmidt_polar()/4
    creations=(model.gammas[:8]-1j*model.gammas[8:])/2
    for mask in range(1,256):
        bit=(mask&-mask).bit_length()-1
        U[:,mask]=creations[bit]@U[:,mask^(1<<bit)]
    assert np.array_equal(U.conj().T@U,np.eye(256))
    D=scaled_basis();omega=U.conj().T@((7*D[:,0]-D[:,2])/4)
    assert all(mask.bit_count()==4 for mask in np.flatnonzero(omega))
    assert np.array_equal(omega.real,np.rint(omega.real)) and not omega.imag.any()
    assert np.vdot(omega,omega)==14
    pairs=list(combinations(range(8),2));T=np.zeros((28,28),complex)
    for i,ab in enumerate(pairs):
        for j,cd in enumerate(pairs):
            order=ab+cd
            if len(set(order))<4:continue
            sign=(-1)**sum(order[a]>order[b] for a in range(4) for b in range(a+1,4))
            T[i,j]=sign*omega[sum(1<<a for a in order)].conjugate()
    M=T.conj().T@T
    assert np.array_equal(T,T.T)
    assert np.array_equal((M-np.eye(28))@(M-9*np.eye(28)),np.zeros((28,28)))
    assert np.trace(M)==84
    return dict(exact_hopping_Fock_basis_unitarity=True,
        exact_integer_Cayley_coefficients={str(i):int(omega[i].real) for i in np.flatnonzero(omega)},
        exact_pairing_matrix_singular_values='3 with multiplicity 7, 1 with multiplicity 21',
        exact_projector_polynomial_verified=True,
        four_excitation_amplitude_formula='a0 * z.T @ T @ z / (2 sqrt(14)), z=(Z_ab)_(a<b)',
        window_support_bound='|<e4,g>| <= 3 X / (sqrt(14) sqrt(1-2X)), for X<1/2',
        proof_uses_Gaussianity_not_sampling=True),U,T
