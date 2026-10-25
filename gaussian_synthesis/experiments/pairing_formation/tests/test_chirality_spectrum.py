import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.chirality_spectrum import ChiralitySpectrum


class ChiralitySpectrumTests(unittest.TestCase):
    def test_spectrum_matches_full_physical_pair_operator(self):
        model,_=angular_model(10);probe=ChiralitySpectrum(model);rng=np.random.default_rng(830)
        state=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);state/=np.linalg.norm(state)
        kernel=np.zeros((10,10),complex);kernel[:5,:5]=probe.kernel;kernel[5:,5:]=probe.kernel
        operator=model.pair_operator(kernel);action=operator@state;obs=probe.evaluate(state)
        self.assertAlmostEqual(obs['chi_mean'],np.vdot(state,action).real,places=12)
        self.assertAlmostEqual(obs['chi_second'],np.linalg.norm(action)**2,places=12)
        self.assertAlmostEqual(obs['chi_fourth'],np.linalg.norm(operator@action)**2,places=12)
        self.assertAlmostEqual(sum(obs['histogram_probabilities']),1.,places=12)

    def test_sign_dephasing_preserves_all_transfer_probabilities(self):
        model,_=angular_model(10);probe=ChiralitySpectrum(model);state=model.evolve([1.7])[0]
        parts=probe.dephase_sign_components(state)
        np.testing.assert_allclose(parts.sum(axis=0),state,atol=2e-13)
        gram=parts.conj()@parts.T
        np.testing.assert_allclose(gram,np.diag(np.diag(gram)),atol=2e-13)
        original=np.bincount(model.m,weights=abs(state)**2,minlength=6)
        mixed=np.bincount(model.m,weights=np.sum(abs(parts)**2,axis=0),minlength=6)
        np.testing.assert_allclose(mixed,original,atol=2e-13)
        self.assertAlmostEqual(probe.evaluate(state)['reflection_expectation'],1.,places=12)


if __name__=='__main__':unittest.main()
