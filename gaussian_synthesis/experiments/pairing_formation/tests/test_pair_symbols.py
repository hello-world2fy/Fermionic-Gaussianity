import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import unittest
from itertools import product,combinations_with_replacement,permutations
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.full_cartan import FullEvenCartan2
from pairing_formation.pair_symbols import PairSymbol2
from pairing_formation.model import picket_fence


def physical(mask,L):return sum(3<<(2*j) for j in range(L) if mask&(1<<j))


class PairSymbolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=picket_fence(4,2);cls.backend=PairSymbol2(cls.model,device='cpu');cls.full=FullEvenCartan2(8)
        rng=np.random.default_rng(1703);cls.u=rng.normal(size=cls.model.dim)+1j*rng.normal(size=cls.model.dim);cls.u/=np.linalg.norm(cls.u)
        cls.indices=np.array([cls.full.index[physical(m,4)] for m in cls.model.masks])
        cls.uf=np.zeros(cls.full.dim,complex);cls.uf[cls.indices]=cls.u
        cls.S=cls.full.project(np.outer(cls.uf,cls.uf));cls.backend.prepare(cls.u)

    def test_number_changing_and_broken_pair_intermediates(self):
        b=self.backend;f=self.full
        lower=[f.index[physical(m,4)] for m in b.lower];upper=[f.index[physical(m,4)] for m in b.upper]
        np.testing.assert_allclose(b.shifted,self.S[np.ix_(lower,upper)],atol=2e-14)
        for i,j in ((0,1),(0,3),(2,0)):
            masks,broken=b.broken_matrix(self.u,i,j)
            first=[f.index[physical(m,4)|(1<<(2*i))|(1<<(2*j+1))] for m in masks]
            second=[f.index[physical(m,4)|(1<<(2*i+1))|(1<<(2*j))] for m in masks]
            np.testing.assert_allclose(broken,self.S[np.ix_(first,second)],atol=2e-14)

    def test_all_pair_symbols_against_full_fock_quartic_contractions(self):
        f=self.full;b=self.backend;u=self.uf;S=self.S;C=S@u.conj();n=f.n
        for i in range(4):
            for j in range(4):
                if i==j:
                    q1=f.majorana_word((4*i,4*i+1),1j)
                    q2=f.majorana_word((4*i+2,4*i+3),1j)
                    first=(q1@C+S@(q1@u).conj()+q2@C+S@(q2@u).conj())/(2*n)
                    fourth=f.quartic_symbol(u,(4*i,4*i+1,4*i+2,4*i+3),projected=S)
                    expected=(C+first+fourth)/4
                else:
                    expected=np.zeros(f.dim,complex)
                    modes=(2*i,2*i+1,2*j+1,2*j);creation=(True,True,False,False)
                    for imaginary in product((0,1),repeat=4):
                        word=tuple(2*m+a for m,a in zip(modes,imaginary));coefficient=1/16
                        for a,dagger in zip(imaginary,creation):
                            if a:coefficient*=(-1j if dagger else 1j)
                        expected-=coefficient*f.quartic_symbol(u,word,projected=S)
                np.testing.assert_allclose(b.symbol(i,j),expected[self.indices],atol=3e-14,err_msg=f'{i},{j}')
                outside=expected.copy();outside[self.indices]=0
                self.assertLess(np.linalg.norm(outside),2e-14)

    def test_contrast_vanishes_for_all_paired_slater_configurations(self):
        b=PairSymbol2(self.model,device='cpu');n=2*self.model.L
        for basis in range(self.model.dim):
            u=np.zeros(self.model.dim,complex);u[basis]=1;b.prepare(u)
            for i,j in ((0,2),(3,1)):
                parts=b.symbol_parts(i,j)
                np.testing.assert_allclose(parts['anomalous'],parts['full']/3,atol=2e-15)
                dst,src=self.model.transitions[i,j];expected=np.zeros_like(u)
                expected[dst]=3*u[src]/(2*n*(2*n-2))
                np.testing.assert_allclose(parts['full'],expected,atol=2e-15)

    def test_pair_field_product_against_independent_fourth_replica(self):
        # Four physical modes: P4 is the fully symmetric intersection of
        # pairwise Cartan ranges. This does not use the product-symbol formula.
        f=FullEvenCartan2(4);D=f.dim;P2=np.zeros((D*D,D*D))
        for a,b,K in f.blocks:
            ind=a*D+b;P2[np.ix_(ind,ind)]=K
        multisets=list(combinations_with_replacement(range(D),4));E=np.zeros((D**4,len(multisets)))
        for col,m in enumerate(multisets):
            words=sorted(set(permutations(m)))
            for word in words:E[np.ravel_multi_index(word,(D,D,D,D)),col]=1/np.sqrt(len(words))
        projected=(P2@E.reshape(D*D,-1)).reshape(E.shape)
        values,V=np.linalg.eigh(E.T@projected);K=E@V[:,values>1-1e-10]
        self.assertEqual(K.shape[1],294);K=K.reshape(D,D,D,D,-1)
        masks=np.array([1,2]);model=SimpleNamespace(L=2,r=1,dim=2,masks=masks,index={1:0,2:1},occ=np.array([[1.,0.],[0.,1.]]))
        b=PairSymbol2(model,device='cpu');u=np.array([.6,.8j]);b.prepare(u)
        indices=np.array([f.index[physical(m,2)] for m in masks]);uf=np.zeros(D,complex);uf[indices]=u
        contracted=np.einsum('ijeft,i,j->eft',K.conj(),uf,uf)
        pair_plus=[]
        for site in (0,1):
            O=np.zeros((D,D),complex)
            for x,y in product((0,1),repeat=2):
                coeff=.25*(-1j)**(x+y)
                O+=coeff*f.majorana_word((4*site+x,4*site+2+y)).toarray()
            pair_plus.append(O)
        for i,j in product(range(2),repeat=2):
            Q1=pair_plus[i];Q2=pair_plus[j].conj().T
            moved=np.einsum('ec,fd,eft->cdt',Q1,Q2,contracted)
            matrix=np.einsum('abcdt,cdt->ab',K,moved)*(35/294)
            expected=matrix@uf.conj()
            np.testing.assert_allclose(b.symbol_parts(i,j)['anomalous'],expected[indices],atol=3e-14)


if __name__=='__main__':unittest.main()
