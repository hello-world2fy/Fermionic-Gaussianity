import unittest
import numpy as np
from scipy.linalg import qr
from pairing_formation.coherent_response import response_modes, ordered_krylov


class CoherentResponseTests(unittest.TestCase):
    def test_basis_independent_projectors(self):
        rng=np.random.default_rng(914)
        def random(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
        x=random((12,12));A=x@x.conj().T
        x=random((12,12));B=x+x.T
        base=qr(random((12,3)),mode='economic')[0]
        U=qr(random((3,3)))[0]
        one,values=response_modes(A,B,base)
        two,values2=response_modes(A,B,base@U)
        for key in one:
            np.testing.assert_allclose(values[key],values2[key],atol=1e-11)
            for n in (1,2,4):
                a=one[key][:,:n];b=two[key][:,:n]
                np.testing.assert_allclose(a@a.conj().T,b@b.conj().T,atol=2e-12)
                np.testing.assert_allclose(base.conj().T@a,0.,atol=2e-12)

    def test_ordinary_arnoldi_space(self):
        rng=np.random.default_rng(915)
        x=rng.normal(size=(14,14));H=x+x.T
        base=qr(rng.normal(size=(14,2)),mode='economic')[0].astype(complex)
        v=base[:,0].copy();K=ordered_krylov(H,v,base,5)
        V=np.column_stack((base,K))
        np.testing.assert_allclose(V.conj().T@V,np.eye(7),atol=1e-12)
        power=H@v
        for j in range(5):
            sub=np.column_stack((base,K[:,:j+1]))
            projector=sub@sub.conj().T
            self.assertLess(np.linalg.norm(power-projector@power)/np.linalg.norm(power),1e-12)
            power=H@power
