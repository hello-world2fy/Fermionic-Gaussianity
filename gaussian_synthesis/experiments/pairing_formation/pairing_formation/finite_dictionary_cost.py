"""Exact dyadic arithmetic for a specified stored finite dictionary.

This certifies linear-algebra statements about the supplied binary64 columns,
not their ideal Gaussian antecedents and not the full Gaussian dictionary.
"""
from fractions import Fraction
from math import isqrt
import numpy as np


def dyadic(array):
    a=np.asarray(array,complex)
    if a.ndim==1:a=a[:,None]
    ratios=[[float(x).as_integer_ratio() for x in part.ravel()] for part in (a.real,a.imag)]
    exponent=max(den.bit_length()-1 for group in ratios for _,den in group)
    integers=[np.array([num<<(exponent-den.bit_length()+1) for num,den in group],dtype=object).reshape(a.shape) for group in ratios]
    return (*integers,1<<exponent)


def inner(a,b):
    ar,ai,ad=a;br,bi,bd=b
    return ar.T@br+ai.T@bi,ar.T@bi-ai.T@br,ad*bd


def sqrt_upper(x,bits=128):
    x=Fraction(x)
    if x==0:return x
    n=isqrt((x.numerator<<(2*bits))//x.denominator)
    return Fraction(n+1,1<<bits)


def encode(x):
    x=Fraction(x)
    return {'numerator':str(x.numerator),'denominator':str(x.denominator),'float':float(x)}


class FiniteDictionaryCertificate:
    def __init__(self,atoms,free_basis,state):
        self.A=dyadic(atoms);self.B=dyadic(free_basis);self.state=dyadic(state)
        gr,gi,den=inner(self.B,self.B)
        size=len(gr);rho=max(sum(abs(int(gr[i,j])-(den if i==j else 0))+abs(int(gi[i,j])) for j in range(size)) for i in range(size))
        self.rho=Fraction(rho,den)
        if self.rho>=1:raise ValueError('Free basis must have a Gram matrix within unit operator distance of identity')
        ar,ai,ad=self.A
        self.atom_norm=sqrt_upper(Fraction(max(np.sum(ar*ar+ai*ai,axis=0)),ad*ad))
        sr,si,sd=self.state
        self.state_norm=sqrt_upper(Fraction(int(np.sum(sr*sr+si*si)),sd*sd))

    def evaluate(self,witness,error=Fraction(0)):
        w=dyadic(witness);br,bi,bd=inner(self.B,w)
        # B(B*B)^-1 B*w removes the free-space component exactly. Bound its
        # norm using ||B|| <= 1+rho and ||(B*B)^-1|| <= 1/(1-rho).
        leakage=Fraction(sum(abs(int(x)) for x in br.ravel())+sum(abs(int(x)) for x in bi.ravel()),bd)
        correction=(1+self.rho)/(1-self.rho)*leakage
        cr,ci,cd=inner(w,self.A)
        support=max(sqrt_upper(Fraction(int(a)**2+int(b)**2,cd*cd)) for a,b in zip(cr.ravel(),ci.ravel()))
        tr,ti,td=inner(w,self.state);real_overlap=Fraction(int(tr[0,0]),td)
        wr,wi,wd=w;norm=sqrt_upper(Fraction(int(np.sum(wr*wr+wi*wi)),wd*wd))
        numerator=max(Fraction(0),real_overlap-correction*self.state_norm-Fraction(error)*(norm+correction))
        denominator=support+correction*self.atom_norm
        lower=numerator/denominator if denominator else Fraction(0)
        return {'error_allowance':encode(error),'coefficient_l1_lower':encode(lower),
          'projected_witness_overlap_minus_error_lower':encode(numerator),'projected_witness_support_upper':encode(denominator),
          'free_space_projection_correction_norm_upper':encode(correction),'free_basis_Gram_deviation_upper':encode(self.rho),
          'scope':'Exact for supplied stored dyadic columns and state, with free source basis; no certification of ideal Gaussian generation or of the full Gaussian dictionary.'}
