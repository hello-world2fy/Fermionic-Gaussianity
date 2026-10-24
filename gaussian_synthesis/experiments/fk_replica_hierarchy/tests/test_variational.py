import unittest
import numpy as np
from fk4 import paths
from fk4.variational import perturbed_hamiltonian,predict,coherent_and_incoherent_metrics,optimize_gaussian_energy
from fk.model import build_model
from fk.gaussian import sample_gaussian
from fk.observables import exact_metrics


class VariationalTests(unittest.TestCase):
    def test_frozen_generalized_eigenproblem_and_coherent_observables(self):
        model=build_model()
        g=sample_gaussian(8,12,np.random.default_rng(29101)).kets
        H=perturbed_hamiltonian(model,11,.1)
        report,data=predict(g,H)
        Q,_=np.linalg.qr(g.T)
        exact=np.linalg.eigvalsh(Q.conj().T@H@Q)[0]
        self.assertAlmostEqual(report['energy'],exact,delta=1e-10)
        self.assertLess(report['generalized_eigen_residual'],1e-10)
        metrics,kernels=coherent_and_incoherent_metrics(g,data['coefficients'],H,model)
        direct=exact_metrics(data['state'],model)
        self.assertAlmostEqual(metrics['coherent']['wick_deviation'],direct['wick_deviation'],delta=1e-11)
        self.assertAlmostEqual(metrics['coherent']['energy'],report['energy'],delta=1e-11)
        self.assertFalse(report['uses_test_target'])

    def test_perturbation_and_rank_deficient_dictionary(self):
        model=build_model()
        H=perturbed_hamiltonian(model,20,0)
        np.testing.assert_array_equal(H,model.full_hamiltonian(20)[np.ix_(model.even_masks,model.even_masks)])
        g=sample_gaussian(8,1,np.random.default_rng(29102)).kets
        report,data=predict(np.repeat(g,4,axis=0),H)
        self.assertEqual(report['actual_gaussian_terms'],4)
        self.assertEqual(report['retained_rank'],1)
        self.assertAlmostEqual(report['energy'],np.vdot(g[0],H@g[0]).real,delta=1e-11)

    def test_full_gaussian_hfb_energy_at_uniform_free_endpoint(self):
        model=build_model()
        H=perturbed_hamiltonian(model,20,0)
        starts=sample_gaussian(8,4,np.random.default_rng(29103)).kets
        g,report=optimize_gaussian_energy(H,starts,sweeps=50)
        self.assertGreaterEqual(float(min(report['energies'])),-94-1e-10)
        self.assertLess(float(min(report['energies'])),-94+1e-7)
        self.assertTrue(np.all(np.diff(report['energy_history'],axis=0)<1e-9))


if __name__=='__main__':
    unittest.main()
