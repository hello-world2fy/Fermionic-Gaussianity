"""Matrix-free equal-channel flat-shell Hamiltonian in transfer blocks.

The energy is shifted by the Fermi-sea expectation. A maximum transfer
cutoff is exact for specified short-time coefficients only when all paths
that can reach the target sector in that order stay below the cutoff.
"""
from itertools import combinations
from math import comb
import numpy as np
from .pairing_channels import annihilator


class BandTransfer:
    def __init__(self, r, G=4., maximum=None, shell_gap=1.):
        self.r, self.G, self.gap = r, G, shell_gap
        self.maximum = r if maximum is None else min(maximum, r)
        phase = 2*np.pi*np.arange(r)/r
        self.forms = np.column_stack((np.sqrt(2)*np.cos(phase), np.sqrt(2)*np.sin(phase)))
        self.subsets = [list(combinations(range(r),m)) for m in range(self.maximum+1)]
        self.lowering = [None]
        self.K = [None]
        self.dark = [np.zeros((1,1))]
        self.dark_dimensions = [0]
        self.recursion = [None]
        for m in range(1,self.maximum+1):
            operators = [annihilator(r,m,f/np.sqrt(r)) for f in self.forms.T]
            self.lowering.append(operators)
            self.K.append(sum(L.conj().T@L for L in operators).tocsr())
            # Only obtain dark projectors when needed; large near-half-filled
            # intermediate blocks are allowed without a dense SVD.
            self.dark.append(None);self.dark_dimensions.append(None)
            previous = {s:i for i,s in enumerate(self.subsets[m-1])}
            subsets = self.subsets[m]
            first = np.array([s[0] for s in subsets])
            particle_parent = np.array([previous[s[1:]] for s in subsets])
            holes = np.array(subsets)
            hole_parent = np.array([[previous[s[:j]+s[j+1:]] for j in range(m)] for s in subsets])
            self.recursion.append((first,particle_parent,holes,hole_parent))

    def dark_projector(self,m):
        if self.dark[m] is None:
            stack = np.concatenate([L.toarray() for L in self.lowering[m]],axis=0)
            _,s,vh = np.linalg.svd(stack,full_matrices=True)
            rank = int(sum(s>1e-11*max(s[0],1.)))
            basis = vh[rank:].conj().T
            self.dark[m] = basis@basis.conj().T
            self.dark_dimensions[m] = basis.shape[1]
        return self.dark[m]

    def zero(self,dtype=complex):
        return [np.zeros((comb(self.r,m),)*2,dtype=dtype) for m in range(self.maximum+1)]

    def initial(self):
        result=self.zero();result[0][0,0]=1.;return result

    def apply(self,blocks):
        answer=self.zero(dtype=np.result_type(*[x.dtype for x in blocks]))
        for m,C in enumerate(blocks):
            answer[m] += (2*self.gap+2*self.G/self.r)*m*C
            if m:
                K=self.K[m]
                answer[m] -= self.G/2*(K@C+(K@C.T).T)
                for L in self.lowering[m]:
                    temporary=L.conj().T@blocks[m-1]
                    answer[m] -= self.G/2*(L.conj().T@temporary.T).T
            if m<self.maximum:
                for L in self.lowering[m+1]:
                    temporary=L@blocks[m+1]
                    answer[m] -= self.G/2*(L@temporary.T).T
        return answer

    def projected_dark(self,C,m,kind='either'):
        D=self.dark_projector(m)
        left,right=D@C,C@D.T
        both=left@D.T
        return both if kind=='both' else left+right-both

    def taylor(self,order):
        coefficients=[self.initial()]
        for n in range(1,order+1):
            coefficients.append([(-1j/n)*x for x in self.apply(coefficients[-1])])
        return coefficients

    def single_cluster_series(self,coefficients,maximum):
        """Permanent lift of the exact intermediate-normalized m1 series."""
        N=len(coefficients)-1
        reference=np.array([x[0][0,0] for x in coefficients])
        singles=[]
        for n in range(N+1):
            term=coefficients[n][1].copy()
            for k in range(n):term-=singles[k]*reference[n-k]
            singles.append(term/reference[0])
        result=[[np.zeros_like(x) for x in self.zero()[:maximum+1]] for _ in range(N+1)]
        result[0][0][0,0]=1.
        for m in range(1,maximum+1):
            first,pp,holes,hp=self.recursion[m]
            for n in range(m,N+1):
                C=result[n][m]
                for k in range(1,n+1):
                    for j in range(m):
                        C+=singles[k][first[:,None],holes[None,:,j]]*result[n-k][m-1][pp[:,None],hp[None,:,j]]
        return result
