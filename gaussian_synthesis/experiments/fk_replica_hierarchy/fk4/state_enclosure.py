"""Enclose the exact FK ground state around its saved binary64 approximation.

The full 256-dimensional spectral gap is checked by verified congruence
inertia, with deliberately coarse IEEE-754 matrix-product error bounds.
The target residual and its normalization use directed 70-digit intervals.
"""
import math
import numpy as np
from mpmath import iv
from .certificates import _c,_mul,_bounds


def verified_gap_inertia(model,t):
    # The predecessor constructs these operators from signed Majoranas using
    # dyadic arithmetic only. Check their exact integer/dyadic structure.
    assert np.all(model.h_int.imag==0)
    assert np.all(model.h_int.real==np.rint(model.h_int.real))
    assert np.all(model.h_hop.real==0)
    assert np.all(2*model.h_hop.imag==np.rint(2*model.h_hop.imag))
    H=model.full_hamiltonian(t)
    phase=np.array([1j if (m&15).bit_count()%2 else 1 for m in range(256)])
    real=phase.conj()[:,None]*H*phase[None,:]
    assert np.all(real.imag==0)
    real=real.real
    np.testing.assert_array_equal(real,real.T)
    eigenvalues,Q=np.linalg.eigh(real)
    n=len(real)
    if np.max(abs(Q))>2:
        raise ArithmeticError('Q entry bound exceeded')
    b=float((eigenvalues[0]+eigenvalues[1])/2)
    shift=real-b*np.eye(n)
    B=Q.T@(shift@Q)
    # For |Qij|<=2 and |Hij-b deltaij|<=M, two length-n products
    # and diagonal subtraction have error <=32 gamma_(2n) n² M.
    # eps is twice unit roundoff, leaving slack for the bound evaluation.
    eps=2.**-52
    gamma=2*n*eps/(1-2*n*eps)
    M=2.**math.ceil(math.log2(np.max(abs(real))+abs(b)+1))
    product_error=32*gamma*n*n*M
    gram_error=32*gamma*n
    orthogonality=np.max(np.sum(abs(Q.T@Q-np.eye(n)),axis=1))+n*gram_error
    if orthogonality>=1:
        raise ArithmeticError('congruence matrix not verified invertible')
    diagonal=np.diag(B)
    radius=np.sum(abs(B),axis=1)-abs(diagonal)+n*product_error
    positive=diagonal>radius
    negative=diagonal< -radius
    if not (np.all(positive|negative) and negative.sum()==1):
        raise ArithmeticError('full-Hamiltonian inertia did not separate a unique ground state')
    return {'full_dimension':n,'verified_eigenvalues_below_separator':1,
            'excited_spectrum_lower_bound':b,'floating_ground_energy':float(eigenvalues[0]),
            'congruence_invertibility_error_upper':float(orthogonality),
            'congruence_entry_roundoff_bound':float(product_error),
            'minimum_signed_diagonal_dominance_margin':float(np.min(abs(diagonal)-radius)),
            'arithmetic_assumption':'IEEE-754 binary64 dot-product error bound; no overflow/underflow; epsilon=2^-52',
            'operator_convention':'Exact integer h_int and half-integer imaginary h_hop; coupling is the supplied exact binary64 t'}


def enclose_ground_state(model,t,target,digits=70):
    inertia=verified_gap_inertia(model,t)
    iv.dps=digits
    state=model.embed(target)
    v=[_c(x) for x in state]
    norm2=sum((r**2+i**2 for r,i in v),iv.mpf(0))
    norm=iv.sqrt(norm2)
    acted=[]
    for row in range(256):
        r,i=iv.mpf(0),iv.mpf(0)
        columns=np.flatnonzero((model.h_int[row]!=0)|(model.h_hop[row]!=0))
        for col in columns:
            h=(iv.mpf(float(model.h_int[row,col].real)),
               iv.mpf(float(t))*iv.mpf(float(model.h_hop[row,col].imag)))
            ar,ai=_mul(h,v[col])
            r+=ar
            i+=ai
        acted.append((r,i))
    energy=sum((r*a+i*b for (r,i),(a,b) in zip(v,acted)),iv.mpf(0))/norm2
    residual2=sum(((a-energy*r)**2+(b-energy*i)**2 for (r,i),(a,b) in zip(v,acted)),iv.mpf(0))/norm2
    residual=iv.sqrt(residual2)
    gap=iv.mpf(inertia['excited_spectrum_lower_bound'])-energy
    if _bounds(gap)[0]<=0:
        raise ArithmeticError('target Rayleigh quotient outside isolated ground-state window')
    # A normalized state has excited weight <=residual²/gap². Select the
    # exact eigenstate phase aligned with the saved target.
    error=abs(norm-1)+iv.sqrt(2)*residual/gap
    return {**inertia,'rayleigh_energy_interval':_bounds(energy),
            'normalized_residual_upper':_bounds(residual)[1],
            'saved_target_to_exact_ground_state_distance_upper':_bounds(error)[1],
            'target_norm_interval':_bounds(norm),'interval_decimal_digits':digits,
            'ground_state_phase':'aligned with the supplied saved target'}
