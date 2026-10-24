"""IID Haar sampling conditional on small quasiparticle number.

The proposal is an enclosing number region, NOT a claim that its atoms lie
in the Hamiltonian energy window. The window indicator remains in integrals.
"""
from functools import lru_cache
from fractions import Fraction
from itertools import combinations
from math import comb,factorial
import numpy as np


@lru_cache(None)
def vandermonde_polynomial(m=4):
    terms={(0,)*m:1}
    for i,j in combinations(range(m),2):
        result={}
        for exponents,c in terms.items():
            for r in range(5):
                e=list(exponents);e[i]+=r;e[j]+=4-r;e=tuple(e)
                result[e]=result.get(e,0)+c*comb(4,r)*(-1)**r
        terms={e:c for e,c in result.items() if c}
    return terms


def polynomial_integral(simplex=True,m=4,extra=None):
    total=Fraction(0)
    extra=(0,)*m if extra is None else tuple(extra)
    for powers,c in vandermonde_polynomial(m).items():
        powers=tuple(a+b for a,b in zip(powers,extra))
        if simplex:
            numerator=c
            for a in powers:numerator*=factorial(a)
            total+=Fraction(numerator,factorial(sum(powers)+m))
        else:
            denominator=1
            for a in powers:denominator*=a+1
            total+=Fraction(c,denominator)
    return total


@lru_cache(None)
def number_volume_constant():
    value=polynomial_integral(True)/polynomial_integral(False)
    if value!=Fraction(1,23178480):
        raise ArithmeticError('Exact number-ball Haar volume mismatch')
    return value


def sample_pair_probabilities(count,cap,rng):
    if not 0<cap<=1:raise ValueError('This exact simplex proposal requires 0<sum p cap<=1')
    # Dumitriu--Edelman beta=4, m=4, a=7: exponent a-(1+beta*(m-1)/2)=0.
    B=np.zeros((count,4,4))
    for j in range(4):B[:,j,j]=np.sqrt(rng.chisquare(14-4*j,count))
    for j in range(3):B[:,j+1,j]=np.sqrt(rng.chisquare(12-4*j,count))
    values=np.linalg.eigvalsh(B@B.transpose(0,2,1))
    fractions=values/values.sum(axis=1)[:,None]
    radius=cap*rng.random(count)**(1/28)
    return fractions*radius[:,None]


def sample_number_ball(count,cap,rng):
    """Actual normalized eight-mode Gaussian kets in their reference Fock frame."""
    p=sample_pair_probabilities(count,cap,rng)
    matrix=rng.normal(size=(count,8,8))+1j*rng.normal(size=(count,8,8))
    U,R=np.linalg.qr(matrix)
    diag=np.diagonal(R,axis1=1,axis2=2)
    U*=np.where(abs(diag)>0,diag/np.maximum(abs(diag),1e-300),1)[:,None,:]
    singular=np.sqrt(p/(1-p))
    J=np.zeros((count,8,8),complex)
    for j in range(4):J[:,2*j,2*j+1]=singular[:,j];J[:,2*j+1,2*j]=-singular[:,j]
    Z=U@J@U.transpose(0,2,1)
    masks=[m for m in range(256) if m.bit_count()%2==0];lookup={m:i for i,m in enumerate(masks)}
    pf=np.zeros((count,128),complex);pf[:,0]=1
    for mask in masks[1:]:
        indices=[i for i in range(8) if mask>>i&1];i=indices[0]
        for pos,j in enumerate(indices[1:]):
            pf[:,lookup[mask]]+=(-1)**pos*Z[:,i,j]*pf[:,lookup[mask^(1<<i)^(1<<j)]]
    kets=pf*np.sqrt(np.prod(1-p,axis=1))[:,None]
    if np.max(abs(np.sum(abs(kets)**2,axis=1)-1))>2e-12:
        raise ArithmeticError('Pairing-chart Gaussian normalization failed')
    return kets,p


def hopping_fock_frame(model):
    from fk5.joint_dual_certificates import exact_schmidt_polar
    U=np.zeros((256,256),complex);U[:,0]=exact_schmidt_polar()/4
    creation=(model.gammas[:8]-1j*model.gammas[8:])/2
    for mask in range(1,256):
        bit=(mask&-mask).bit_length()-1
        U[:,mask]=creation[bit]@U[:,mask^(1<<bit)]
    if not np.array_equal(U.conj().T@U,np.eye(256)):
        raise ArithmeticError('Hopping Fock frame lost exact unitarity')
    return U[np.ix_(model.even_masks,model.even_masks)]
