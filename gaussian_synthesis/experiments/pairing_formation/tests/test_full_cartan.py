import sys
from pathlib import Path
import unittest
from math import comb
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.full_cartan import FullEvenCartan2
from pairing_formation.replica2 import Replica2
from pairing_formation.model import picket_fence


class FullCartanTests(unittest.TestCase):
    def test_full_projector_dimension_idempotence_and_replica_symmetry(self):
        rng=np.random.default_rng(1700)
        for n in (2,3,4,6):
            cartan=FullEvenCartan2(n)
            matrix=rng.normal(size=(cartan.dim,cartan.dim))+1j*rng.normal(size=(cartan.dim,cartan.dim))
            projected=cartan.project(matrix)
            np.testing.assert_allclose(cartan.project(projected),projected,atol=3e-14)
            np.testing.assert_allclose(projected.T,projected,atol=3e-14)
            self.assertAlmostEqual(sum(np.trace(K) for a,b,K in cartan.blocks),comb(2*n,n)/2,places=12)

    def test_independent_paired_beta_kernel_embedding(self):
        model=picket_fence(4,2);paired=Replica2(model);full=FullEvenCartan2(8)
        indices=[full.index[sum(3<<(2*j) for j in range(model.L) if mask&(1<<j))] for mask in model.masks]
        rng=np.random.default_rng(1701);u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);u/=np.linalg.norm(u)
        v=np.zeros(full.dim,complex);v[indices]=u
        q,C=full.evaluate(v);q0,C0=paired.evaluate(u);expected=np.zeros(full.dim,complex);expected[indices]=C0
        self.assertAlmostEqual(q,q0,places=13);np.testing.assert_allclose(C,expected,atol=2e-14)

    def test_majorana_operator_products(self):
        full=FullEvenCartan2(6)
        X=full.majorana_word((0,3),1j);Y=full.majorana_word((5,8),1j)
        O=full.majorana_word((0,3,5,8),-1)
        np.testing.assert_allclose((X@Y).toarray(),O.toarray(),atol=1e-15)
        np.testing.assert_allclose(X.toarray(),X.conj().T.toarray(),atol=1e-15)
        np.testing.assert_allclose((X@X).toarray(),np.eye(full.dim),atol=1e-15)

    def test_quartic_insertion_against_independent_third_replica(self):
        # P3 is built as the intersection of the three full pairwise Cartan
        # ranges. Its partial trace gives the insertion without using W(T).
        rng=np.random.default_rng(1702)
        for n,expected_rank in ((2,4),(3,20),(4,112)):
            cartan=FullEvenCartan2(n);D=cartan.dim;P2=np.zeros((D*D,D*D))
            for a,b,K in cartan.blocks:
                ind=a*D+b;P2[np.ix_(ind,ind)]=K
            P12=np.kron(P2,np.eye(D));perm=np.arange(D**3).reshape(D,D,D).transpose(0,2,1).ravel()
            value,V=np.linalg.eigh(P12+np.kron(np.eye(D),P2)+P12[np.ix_(perm,perm)])
            Q=V[:,value>3-1e-10];self.assertEqual(Q.shape[1],expected_rank);P3=Q@Q.T
            u=rng.normal(size=D)+1j*rng.normal(size=D);u/=np.linalg.norm(u)
            for indices in ((0,1,2,3),(0,2,4,6)):
                if max(indices)>=2*n:continue
                O=cartan.majorana_word(indices,-1).toarray()
                insertion=np.einsum('fc,acbf->ab',O,P3.reshape(D*D,D,D*D,D))*comb(2*n,n)/(2*expected_rank)
                direct=(insertion@np.outer(u,u).ravel()).reshape(D,D)@u.conj()
                np.testing.assert_allclose(cartan.quartic_symbol(u,indices),direct,atol=2e-14)


if __name__=='__main__':unittest.main()
