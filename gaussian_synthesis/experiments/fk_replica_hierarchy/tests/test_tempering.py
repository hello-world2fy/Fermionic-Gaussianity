import unittest
import numpy as np
from scipy.special import logsumexp

from fk4 import paths
from fk4.tempering import heatbath, sample_tempered, next_temperature, incremental_ess
from fk4.reference import log_dimension, magic_log_z, magic_reference, gaussian_reference
from fk4.weighted import WeightedMoments
from fk.gaussian import sample_gaussian
from fk.moments import ProjectedMoments
from fk3.discovery import covariance_from_kets
from fk4.solver import solve
from fk.model import build_model
from fk2.exact_moments import ExactK2Moments, full_tensor_reconstruction


class TemperingTests(unittest.TestCase):
    def test_newton_roundoff_safeguard_full_tensor_regression(self):
        model = build_model()
        backend = ExactK2Moments(model)
        for t in (0, 5, 9, 10, 10.9225, 11, 12, 20):
            target = model.target(t)
            solution = solve(backend, model.S.conj().T@target)
            self.assertTrue(solution["converged"])
            full = full_tensor_reconstruction(model.S@solution["z"])
            self.assertLess(np.linalg.norm(full-target), 1e-10)

    def test_weighted_backend_matches_independent_haar_backend_and_derivatives(self):
        rng = np.random.default_rng(71)
        h = rng.normal(size=(400, 3))+1j*rng.normal(size=(400, 3))
        z = np.array([.12+.08j, -.1+.2j, .17-.09j])
        for k in (2, 4, 16):
            exact = ProjectedMoments(h, k, 2.).evaluate(z)
            backend = WeightedMoments(h, np.full(len(h), -np.log(len(h))), k, 2.)
            value = backend.evaluate(z)
            np.testing.assert_allclose(value.q, exact.q, rtol=1e-12)
            np.testing.assert_allclose(value.gradient, exact.gradient, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(value.hessian, exact.hessian, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(value.mass, exact.mass, rtol=1e-12)

    def test_heatbath_preserves_gaussianity_and_haar_second_moment(self):
        rng = np.random.default_rng(81)
        states = sample_gaussian(4, 4096, rng).kets.copy()
        witness = np.zeros(8, complex)
        witness[0] = 1
        heatbath(states, witness, 0., rng, n=4)
        np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1., atol=1e-13)
        cov = covariance_from_kets(states[:32], n=4)
        np.testing.assert_allclose(cov@cov, -np.eye(8)[None]*np.ones((32, 1, 1)), atol=1e-12)
        frame = states.conj().T@states/len(states)
        self.assertLess(np.linalg.norm(frame-np.eye(8)/8), .025)

    def test_magic_tempering_independent_analytic_moment(self):
        magic = np.zeros(8, complex)
        magic[[0, 7]] = 1/np.sqrt(2)
        result = sample_tempered(magic, 8, 4096, np.random.default_rng(761), n=4, sweeps=2)
        self.assertLess(abs(result.log_z-magic_log_z(8)), .09)
        # Under pi_beta, 2|<M|g>|^2 ~ Beta(beta+1, 3).
        r = 2*np.exp(result.log_r)
        self.assertLess(abs(r.mean()-9/12), .015)

    def test_gaussian_tempering_independent_partition_and_overlap_mean(self):
        vacuum = np.zeros(8, complex)
        vacuum[0] = 1
        result = sample_tempered(vacuum, 4, 4096, np.random.default_rng(981), n=4, sweeps=2)
        self.assertLess(abs(result.log_z+log_dimension(4, 4)), .09)
        reference = np.exp(log_dimension(4, 4)-log_dimension(4, 5))
        self.assertLess(abs(np.exp(result.log_r).mean()-reference), .015)

    def test_temperature_selection_and_reference_finite_k_bias(self):
        log_r = np.linspace(-30, -1, 400)
        beta = next_temperature(log_r, 0, 10, .8)
        self.assertAlmostEqual(incremental_ess(log_r, beta), 320, places=7)
        for k in (1, 2, 4, 8, 16, 32, 64, 128):
            self.assertLess(magic_reference(k)["L"], np.sqrt(2))
            self.assertGreater(magic_reference(k)["B"], np.sqrt(2))
            self.assertEqual(gaussian_reference(8, k)["L"], 1.)
            self.assertGreater(gaussian_reference(8, k)["B"], 1.)


if __name__ == "__main__":
    unittest.main()
