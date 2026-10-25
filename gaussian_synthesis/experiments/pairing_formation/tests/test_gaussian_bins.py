import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.chirality_spectrum import ChiralitySpectrum
from pairing_formation.gaussian_bins import angular_invariant_basis
from scripts.assess_gaussian_bins import stability


class GaussianBinsTests(unittest.TestCase):
    def test_exact_symmetry_and_absolute_circulation_invariance(self):
        model,_=angular_model(10);Q,permutations=angular_invariant_basis(model)
        np.testing.assert_allclose(Q.T@Q,np.eye(Q.shape[1]),atol=2e-14)
        np.testing.assert_allclose(model.H@Q,Q@(Q.T@(model.H@Q)),atol=2e-13)
        np.testing.assert_allclose(Q@(Q.T@model.initial),model.initial,atol=2e-14)
        probe=ChiralitySpectrum(model);K=np.zeros((10,10),complex);K[:5,:5]=probe.kernel;K[5:,5:]=probe.kernel
        chi=model.pair_operator(K);rng=np.random.default_rng(840)
        v=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);v/=np.linalg.norm(v)
        for p in permutations:
            self.assertAlmostEqual(abs(np.vdot(v,chi@v)),abs(np.vdot(v[p],chi@v[p])),places=12)
            self.assertAlmostEqual(abs(v)**2@model.m,abs(v[p])**2@model.m,places=12)

    def test_independent_pool_gate_distinguishes_signal_from_noise(self):
        rng=np.random.default_rng(851)
        full,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
        base=full[:,:2]
        contrasts=np.array([[1.,-1,0,0],[1,1,-1,-1]]).T
        contrasts/=np.linalg.norm(contrasts,axis=0)
        signal=(full[:,2:4]@np.diag([1.,.4])@contrasts.T).T
        # Large shared components must not be reported as added structure.
        shared=(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))@base.T
        noise=.02*(rng.normal(size=(128,4,8))+1j*rng.normal(size=(128,4,8)))
        result,mean,U=stability(signal+shared+noise,base)
        self.assertTrue(next(x for x in result['rank_checks'] if x['rank']==2)['passes_declared_gate'])
        self.assertGreater(np.min(np.linalg.svd(full[:,2:4].conj().T@U[:,:2],compute_uv=False)),.999)
        np.testing.assert_allclose(mean@base.conj(),0,atol=2e-15)
        pure_noise,_,_=stability(shared+noise,base)
        self.assertFalse(any(x['passes_declared_gate'] for x in pure_noise['rank_checks']))


if __name__=='__main__':unittest.main()
