import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import unittest
from itertools import combinations
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.full_cartan import FullEvenCartan2
from pairing_formation.multipair_symbols import (
    MultiPairSymbol2, full_majorana_symbol, even_splits, physical_masks,
    act_fermion_word, pair_word)


class MultiPairSymbolsTests(unittest.TestCase):
    def test_stochastic_square_symbol_against_full_spin_product(self):
        from pairing_formation.multipair_noise import distinct_transfer_square_symbol
        rng=np.random.default_rng(1912);r=2;L=4;plus=np.array([[0,0],[1,0]],complex)
        operators=[]
        for site in range(L):
            op=np.array([[1.]])
            for j in range(L):op=np.kron(op,plus if j==site else np.eye(2))
            operators.append(op)
        for _ in range(8):
            local=rng.normal(size=(L,2))+1j*rng.normal(size=(L,2));local/=np.linalg.norm(local,axis=1)[:,None]
            state=np.array([1.+0j])
            for v in local:state=np.kron(state,v)
            W=rng.normal(size=(r,r));X=sum(W[i,j]*operators[r+i]@operators[j].conj().T for i in range(r) for j in range(r))
            p=local[r:,1].conj()*local[r:,0];h=local[:r,0].conj()*local[:r,1]
            np.testing.assert_allclose(distinct_transfer_square_symbol(p,h,W),np.vdot(state,X@X@state),atol=2e-14)

    def test_cross_copy_remainder_vanishes_on_paired_slater_basis(self):
        model=picket_fence(4,2);b=MultiPairSymbol2(model,device='cpu')
        for j in range(model.dim):
            u=np.eye(model.dim,dtype=complex)[j];b.prepare(u)
            full,terms=b.monomial_symbol((2,3),(0,1),True)
            np.testing.assert_allclose(full,terms['(0, 0)']+terms['(8, 0)'],atol=2e-15)
            chosen,masks,values=act_fermion_word(physical_masks(model.masks,4),pair_word((2,3),(0,1)))
            expected=np.zeros_like(u);lookup={int(m):i for i,m in enumerate(physical_masks(model.masks,4))}
            expected[[lookup[int(m)] for m in masks]]=105*values*u[chosen]/(16*14*12*10)
            np.testing.assert_allclose(full,expected,atol=2e-15)

    def test_parity_twirl_readout_is_exact_double_commutator(self):
        from pairing_formation.multipair_noise import transfer_operators,noise_observables
        from pairing_formation.pairing_channels import angular_model
        model,_=angular_model(6);rng=np.random.default_rng(1903)
        u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);u/=np.linalg.norm(u)
        parity=np.column_stack([u*(model.m%2==p) for p in (0,1)])
        counts=np.column_stack([u*(model.m==p) for p in range(model.r+1)])
        delta=parity@parity.conj().T-counts@counts.conj().T
        np.testing.assert_allclose(np.diag(delta),0,atol=1e-15)
        for i,j in combinations(range(model.L),2):
            W=np.zeros((model.L,model.L));W[i,j]=1
            self.assertLess(abs(np.trace(delta@model.pair_operator(W).toarray())),2e-15)
        N=np.diag(model.m);N2=N@N
        for X in transfer_operators(model):
            X=X.toarray();second=np.vdot(u,X@X@u)
            for phi in (0.,np.pi/4):
                Q=np.exp(-1j*phi)*X+np.exp(1j*phi)*X.conj().T
                comm=Q@N2-N2@Q
                coefficient=-.5*np.trace(delta@(Q@comm-comm@Q))
                np.testing.assert_allclose(coefficient,-2*np.real(np.exp(-2j*phi)*second),atol=1e-14)
        operators=transfer_operators(model);means,connected,variance=noise_observables(operators,u)
        np.testing.assert_allclose(variance[::2]-variance[1::2],4*np.diag(connected).real,atol=2e-14)

    def test_higher_exterior_symbols_against_clifford_duality(self):
        rng = np.random.default_rng(1900)
        for n, m in ((4, 3), (4, 4), (5, 3), (6, 4)):
            f = FullEvenCartan2(n)
            u = rng.normal(size=f.dim)+1j*rng.normal(size=f.dim)
            u /= np.linalg.norm(u)
            S = f.project(np.outer(u, u)); C = S@u.conj()
            for _ in range(3):
                indices = tuple(rng.choice(2*n, 2*m, replace=False))
                complement = tuple(j for j in range(2*n) if j not in indices)
                O = f.majorana_word(indices, 1j**m).toarray()
                Oc = f.majorana_word(complement, 1j**(n-m)).toarray()
                sign = np.vdot(Oc, O)/f.dim
                np.testing.assert_allclose(O, sign*Oc, atol=1e-14)
                if len(complement) == 0:
                    expected = C
                elif len(complement) == 2:
                    expected = (Oc@C + S@(Oc@u).conj())/(2*n)
                else:
                    expected = f.quartic_symbol(u, complement, projected=S)
                np.testing.assert_allclose(full_majorana_symbol(f, u, indices, S), sign*expected, atol=3e-14)

    def test_all_intermediate_sector_types_against_full_fock(self):
        model = picket_fence(4, 2); b = MultiPairSymbol2(model, device='cpu')
        f = FullEvenCartan2(8);rng = np.random.default_rng(1901)
        u = rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);u/=np.linalg.norm(u)
        ind = [f.index[int(m)] for m in physical_masks(model.masks, 4)]
        uf=np.zeros(f.dim,complex);uf[ind]=u;S=f.project(np.outer(uf,uf));b.prepare(u)
        cases=[((),0),((),1),((),2),((0,2),0),((0,2),1),((0,2),2),((0,1,2,3),0)]
        for broken,number in cases:
            left,right,matrix=b.sector_matrix(broken,number)
            lm=physical_masks(left,4)|sum(1<<(2*j) for j in broken)
            rm=physical_masks(right,4)|sum(1<<(2*j+1) for j in broken)
            li=[f.index[int(m)] for m in lm];ri=[f.index[int(m)] for m in rm]
            np.testing.assert_allclose(matrix,S[np.ix_(li,ri)],atol=2e-14)

    def test_disjoint_pair_symbols_against_direct_full_fock_coproduct(self):
        model=picket_fence(4,2);b=MultiPairSymbol2(model,device='cpu');f=FullEvenCartan2(8)
        rng=np.random.default_rng(1902);u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);u/=np.linalg.norm(u)
        ind=np.array([f.index[int(m)] for m in physical_masks(model.masks,4)])
        uf=np.zeros(f.dim,complex);uf[ind]=u;S=f.project(np.outer(uf,uf));b.prepare(u)
        from scipy.sparse import coo_matrix
        def operator(word):
            chosen,masks,values=act_fermion_word(f.masks,word)
            return coo_matrix((values,([f.index[int(m)] for m in masks],chosen)),shape=(f.dim,f.dim)).tocsr()
        for plus,minus in (((3,),(0,)),((0,2),(1,3)),((3,1),(2,0))):
            word=pair_word(plus,minus);expected=np.zeros(f.dim,complex)
            for left,right,coefficient in even_splits(len(word)):
                X=operator(tuple(word[i] for i in left));Y=operator(tuple(word[i] for i in right))
                expected+=coefficient*(X@(S@(Y.conj().T@uf).conj()))
            denominator=np.prod([2*f.n-2*j for j in range(len(word)//2)])
            expected/=denominator
            np.testing.assert_allclose(b.monomial_symbol(plus,minus),expected[ind],atol=3e-14)
            if len(plus)==1:
                np.testing.assert_allclose(b.monomial_symbol(plus,minus),b.symbol(plus[0],minus[0]),atol=3e-14)


if __name__=='__main__':unittest.main()
