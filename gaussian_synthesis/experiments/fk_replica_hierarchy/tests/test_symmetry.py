import unittest
import numpy as np
from fk4 import paths
from fk4.symmetry import time_reverse, TimeReversalAveraged
from fk4.weighted import WeightedMoments
from fk.model import build_model
from fk.gaussian import sample_gaussian
from fk3.discovery import covariance_from_kets


class SymmetryTests(unittest.TestCase):
    def test_antiunitary_fixed_basis_and_gaussian_preservation(self):
        model = build_model()
        np.testing.assert_allclose(time_reverse(model.S.T), model.S.T, atol=1e-15)
        for t in (0, 9, 10.9225, 20):
            np.testing.assert_allclose(time_reverse(model.target(t)), model.target(t), atol=1e-15)
        states = sample_gaussian(8, 16, np.random.default_rng(351)).kets
        cov = covariance_from_kets(time_reverse(states))
        np.testing.assert_allclose(cov@cov, -np.ones((16, 1, 1))*np.eye(16)[None], atol=1e-12)

    def test_symmetrization_agrees_with_explicit_paired_quadrature(self):
        model = build_model()
        rng = np.random.default_rng(352)
        states = sample_gaussian(8, 32, rng).kets
        h = states.conj()@model.S
        weights = rng.normal(size=len(h))-np.log(len(h))
        first = TimeReversalAveraged(WeightedMoments(h, weights, 4, 2.))
        paired = np.concatenate((states, time_reverse(states)))
        second = WeightedMoments(paired.conj()@model.S, np.tile(weights-np.log(2), 2), 4, 2.)
        z = rng.normal(size=3)+1j*rng.normal(size=3)
        a, b = first.evaluate(z), second.evaluate(z)
        np.testing.assert_allclose(a.q, b.q, rtol=1e-13)
        np.testing.assert_allclose(a.gradient, b.gradient, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(a.hessian, b.hessian, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(first.evaluate(z.real).gradient[3:], 0., atol=1e-15)


if __name__ == "__main__":
    unittest.main()
