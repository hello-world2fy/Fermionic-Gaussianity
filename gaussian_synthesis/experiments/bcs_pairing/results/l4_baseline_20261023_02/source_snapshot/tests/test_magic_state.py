import unittest
import numpy as np
from gaussian_synthesis import FockBasis
from gaussian_synthesis.sampling import HaarFourModeSampler
from gaussian_synthesis.moments import MomentAccumulator, batch_moments
from gaussian_synthesis.theory import dimension, log_second_moment_bound
from _magic_reference import magic_state, magic_q, magic_optimizer, magic_moment, magic_mass


class MagicStateTests(unittest.TestCase):
    def test_analytic_values_and_scaling(self):
        self.assertEqual([dimension(4, k) for k in (1, 2, 4)], [8, 35, 294])
        np.testing.assert_allclose([magic_q(k) for k in (1, 2, 4)], [1, .875, .525])
        for k in (1, 2, 4):
            u = magic_optimizer(k, FockBasis())
            scale = np.linalg.norm(u)
            self.assertAlmostEqual(scale**(2*k-1)*magic_q(k), 1)
            self.assertAlmostEqual(magic_mass(k), magic_moment(k-.5)/magic_moment(k))
            actual_second = dimension(4, k)**2*scale**(4*k-2)*magic_moment(2*k-1)
            self.assertLessEqual(actual_second, np.exp(log_second_moment_bound(4, k, scale)))

    def test_moments_against_analytic_distribution(self):
        sampler = HaarFourModeSampler(np.random.default_rng(79))
        magic = magic_state(FockBasis())
        accumulators = {k: MomentAccumulator(8) for k in (1, 2, 4)}
        for _ in range(4):
            g = sampler.sample(8192)
            for k, accumulator in accumulators.items():
                accumulator.add(batch_moments(magic, k, g))
        for k, accumulator in accumulators.items():
            estimate = accumulator.result()
            q_se = dimension(4, k)*np.sqrt((magic_moment(2*k)-magic_moment(k)**2)/estimate.samples)
            self.assertLess(abs(estimate.q-magic_q(k)), 6*q_se)
            # This checks the entire complex vector, not only its magic-axis component.
            expected_C = magic_q(k)*magic
            variance_bound = dimension(4, k)**2*magic_moment(2*k-1)
            self.assertLess(np.linalg.norm(estimate.C-expected_C), 5*np.sqrt(variance_bound/estimate.samples))
