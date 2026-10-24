"""Independent interval inertia in small full-Fock connected components.

This differs from the predecessor's global binary64 congruence error model:
every product in each <=18-dimensional congruence is evaluated with directed
intervals. Double-precision eigenvectors only choose invertible coordinates.
"""
import numpy as np
from fractions import Fraction
from scipy.sparse.csgraph import connected_components
from mpmath import iv
from . import paths
from fk4.certificates import _bounds


def spectral_certificate(model,t,digits=70):
    iv.dps=digits
    phase=np.array([1j if (m&15).bit_count()%2 else 1 for m in range(256)])
    A0=phase.conj()[:,None]*model.h_int*phase[None,:]
    A1=phase.conj()[:,None]*model.h_hop*phase[None,:]
    assert not A0.imag.any() and not A1.imag.any()
    assert np.array_equal(A0.real,np.rint(A0.real))
    assert np.array_equal(A1.real*2,np.rint(A1.real*2))
    count,labels=connected_components((abs(A0)+abs(A1))>0,directed=False)
    blocks=[];floating=[]
    for b in range(count):
        ids=np.flatnonzero(labels==b);n=len(ids)
        a=A0.real[np.ix_(ids,ids)];v=A1.real[np.ix_(ids,ids)]
        vals,Q=np.linalg.eigh(a+float(t)*v);floating.extend(vals.tolist())
        qi=iv.matrix([[iv.mpf(float(Q[i,j])) for j in range(n)] for i in range(n)])
        ai=iv.matrix([[iv.mpf(float(a[i,j]))+iv.mpf(float(t))*iv.mpf(float(v[i,j])) for j in range(n)] for i in range(n)])
        gram=qi.T*qi;congruence=qi.T*ai*qi
        for i in range(n):
            radius=sum((abs(gram[i,j]) for j in range(n) if i!=j),iv.mpf(0))
            assert _bounds(gram[i,i]-radius)[0]>0
        blocks.append((congruence,gram,n))
    floating.sort();margin=1e-8
    lower=float(np.nextafter(floating[0]-margin,-np.inf))
    upper=float(np.nextafter(floating[0]+margin,np.inf))
    excited=float(np.nextafter(floating[1]-margin,-np.inf))
    def inertia(separator):
        negatives=0
        for C,G,n in blocks:
            B=C-iv.mpf(separator)*G
            for i in range(n):
                radius=sum((abs(B[i,j]) for j in range(n) if i!=j),iv.mpf(0))
                if _bounds(B[i,i]-radius)[0]>0:continue
                if _bounds(B[i,i]+radius)[1]<0:negatives+=1;continue
                raise ArithmeticError('Interval inertia separator too close to spectrum')
        return negatives
    counts=[inertia(x) for x in [lower,upper,excited]]
    assert counts==[0,1,1]
    return {'ground_energy_interval':[lower,upper],'first_excited_energy_lower':excited,
        'full_spectral_gap_lower':_bounds(iv.mpf(excited)-iv.mpf(upper))[0],
        'full_dimension':256,'connected_components':count,'maximum_block_dimension':max(x[2] for x in blocks),
        'inertia_counts':counts,'interval_decimal_digits':digits,
        'method':'Exact-dyadic Hamiltonian; directed small-block congruence inertia; verified invertible coordinate matrices'}


def exact_rational_spectrum_check(model,t,certificate):
    """Independent exact LDL inertia; no floating congruence eigenvectors."""
    phase=np.array([1j if (m&15).bit_count()%2 else 1 for m in range(256)])
    a=phase.conj()[:,None]*model.h_int*phase[None,:]
    b=phase.conj()[:,None]*model.h_hop*phase[None,:]
    assert not a.imag.any() and not b.imag.any()
    number,labels=connected_components((abs(a)+abs(b))>0,directed=False)
    T=Fraction(float(t));cache={}
    def count(mu):
        negative=0
        for component in range(number):
            ids=np.flatnonzero(labels==component);n=len(ids)
            A=[[Fraction(float(a[i,j].real))+T*Fraction(float(b[i,j].real))-(Fraction(mu) if i==j else 0) for j in ids] for i in ids]
            key=tuple(tuple(row) for row in A)
            if key in cache:negative+=cache[key];continue
            L=[[Fraction(int(i==j)) for j in range(n)] for i in range(n)];D=[]
            for j in range(n):
                d=A[j][j]-sum(L[j][k]**2*D[k] for k in range(j));assert d!=0;D.append(d)
                for i in range(j+1,n):L[i][j]=(A[i][j]-sum(L[i][k]*L[j][k]*D[k] for k in range(j)))/d
            cache[key]=sum(d<0 for d in D);negative+=cache[key]
        return negative
    separators=[*certificate['ground_energy_interval'],certificate['first_excited_energy_lower']]
    actual=[count(x) for x in separators]
    assert actual==[0,1,1]
    exact_gap=Fraction(separators[2])-Fraction(separators[1])
    assert Fraction(certificate['full_spectral_gap_lower'])<=exact_gap
    return {'exact_rational_LDL_counts':actual,'distinct_shifted_blocks_checked':len(cache)}
