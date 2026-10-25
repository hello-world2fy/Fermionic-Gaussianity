"""Small full-even-Fock reference for the fermionic k2 Cartan projector.

No seniority or particle-number projection occurs. The implementation is
deliberately limited to n<=8 physical complex modes and provides independent
checks for symbols whose quadratic factors leave the paired subspace.
"""
from math import comb
import numpy as np
from scipy.sparse import coo_matrix


def zero_fourier_coefficient(size,distance):
    if size==0:return 1.
    if size%2 or distance%2:return 0.
    return (-1.)**(distance//2)*comb(size,size//2)*comb(size//2,distance//2)/comb(size,distance)/2**size


class FullEvenCartan2:
    def __init__(self,physical_modes):
        n=int(physical_modes)
        if not 2<=n<=8:raise ValueError('Full-even reference supports2<=n<=8')
        self.n=n;self.masks=np.array([m for m in range(1<<n) if m.bit_count()%2==0],dtype=np.int64)
        self.dim=len(self.masks);self.index={int(m):i for i,m in enumerate(self.masks)}
        a=np.repeat(np.arange(self.dim),self.dim);b=np.tile(np.arange(self.dim),self.dim)
        ma,mb=self.masks[a],self.masks[b];difference=ma^mb;keys=(ma&mb)|(difference<<n)
        order=np.argsort(keys,kind='stable');boundaries=np.r_[0,np.flatnonzero(np.diff(keys[order]))+1,len(order)]
        self.blocks=[]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            chosen=order[lo:hi];aa=a[chosen];bb=b[chosen];D=int(difference[chosen[0]])
            modes=[j for j in range(n) if D&(1<<j)];s=len(modes)
            alternate=sum(1<<j for j in modes[1::2]);flips=self.masks[aa,None]^self.masks[aa][None,:]
            distance=np.bitwise_count(flips);sign=(-1.)**np.bitwise_count(flips&alternate)
            coeff=np.array([zero_fourier_coefficient(s,d) for d in range(s+1)])
            K=sign*coeff[distance]
            self.blocks.append((aa,bb,K))

    def project(self,matrix):
        output=np.zeros((self.dim,self.dim),complex)
        for a,b,K in self.blocks:output[a,b]=K@matrix[a,b]
        return output

    def evaluate(self,u):
        S=self.project(np.outer(u,u));C=S@u.conj()
        return float(np.vdot(u,C).real),C

    def anomalous(self,u,v):
        return self.project(np.outer(u,u))@v.conj()

    def majorana_word(self,word,factor=1.):
        if len(word)%2:raise ValueError('Even operator word required')
        dst=[];values=[]
        for source in self.masks:
            mask=int(source);value=complex(factor)
            for a in reversed(word):
                j,imaginary=divmod(a,2)
                if j>=self.n:raise ValueError('Majorana index out of range')
                value*=(-1.)**((mask&((1<<j)-1)).bit_count())
                if imaginary:value*=1j*(1-2*((mask>>j)&1))
                mask^=1<<j
            dst.append(self.index[mask]);values.append(value)
        return coo_matrix((values,(dst,np.arange(self.dim))),shape=(self.dim,self.dim)).tocsr()

    def quartic_symbol(self,u,indices,projected=None):
        """Coherent integral for <-gamma_a gamma_b gamma_c gamma_d>.

        Checked against an independent third-replica partial trace and Haar
        integration; includes every quadratic intermediate Fock sector.
        """
        a,b,c,d=indices
        if len(set(indices))!=4:raise ValueError('Four distinct Majoranas required')
        O=self.majorana_word(indices,factor=-1)
        S=self.project(np.outer(u,u)) if projected is None else projected;C=S@u.conj()
        result=3*(O@C+S@(O@u).conj())
        for left,right,sign in (((a,b),(c,d),1),((a,c),(b,d),-1),((a,d),(b,c),1)):
            X=self.majorana_word(left,factor=1j);Y=self.majorana_word(right,factor=1j)
            result+=sign*(X@(S@(Y@u).conj())+Y@(S@(X@u).conj()))
        return result/(2*self.n*(2*self.n-2))
