import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from gaussian_synthesis import FockBasis
from gaussian_synthesis.moments import MomentEstimate
from gaussian_synthesis.solver import ResourceLimitError
from gaussian_synthesis.validation import validate_extent_candidate


class NoDrawSampler:
    basis = FockBasis()

    def sample(self, count):
        raise AssertionError("This validation must not draw samples")


class FixedSampler:
    basis = FockBasis()

    def __init__(self):
        self.draws = 0

    def sample(self, count):
        self.draws += count
        return np.repeat(self.basis.ket(0)[None, :], count, axis=0)


class CountingBudget:
    def __init__(self, stop_at=None):
        self.checks = 0
        self.stop_at = stop_at

    def check(self):
        self.checks += 1
        if self.stop_at is not None and self.checks >= self.stop_at:
            raise ResourceLimitError("Controlled test resource stop")


class ExtentValidationTests(unittest.TestCase):
    def setUp(self):
        self.basis = FockBasis()
        self.psi = (self.basis.ket(0) + self.basis.ket(15)) / np.sqrt(2.)

    @staticmethod
    def fixed_moments(u, k, samples, *, n):
        # q deliberately disagrees: extent validation must use Euler from C.
        return MomentEstimate(999., .875*u, 0., 0., len(samples))

    def test_k1_is_exact_without_draws_and_preserves_input(self):
        u = .7*self.psi
        original = u.copy()
        report = validate_extent_candidate(
            self.psi, u, 1, NoDrawSampler(), CountingBudget())
        self.assertEqual(report.status, "certified")
        self.assertEqual(report.samples, 0)
        self.assertEqual(report.requested_samples, 0)
        self.assertIsNone(report.delta)
        self.assertEqual(report.statistical_radius, 0.)
        self.assertAlmostEqual(report.q_lower, .49)
        self.assertEqual(report.q_lower, report.q_upper)
        self.assertAlmostEqual(report.residual_bound, .3)
        np.testing.assert_array_equal(report.reconstruction, original)
        np.testing.assert_array_equal(u, original)
        self.assertTrue(u.flags.writeable)
        self.assertEqual(report.candidate_sha256,
                         hashlib.sha256(original.tobytes()).hexdigest())
        self.assertNotIn("reconstruction", report.summary())
        json.dumps(report.summary(), allow_nan=False)

    def test_zero_candidate_is_exact_without_draws(self):
        report = validate_extent_candidate(
            self.psi, np.zeros(8), 4, NoDrawSampler(), CountingBudget())
        self.assertEqual(report.status, "certified")
        self.assertEqual(report.q_lower, 0.)
        self.assertEqual(report.q_upper, 0.)
        self.assertEqual(report.samples, 0)
        self.assertIsNone(report.delta)
        self.assertAlmostEqual(report.residual_bound, 1.)

    def test_sample_budget_rejects_before_any_draws(self):
        report = validate_extent_candidate(
            self.psi, self.psi, 4, NoDrawSampler(), CountingBudget(),
            max_samples=1)
        self.assertEqual(report.status, "sample_budget_exceeded")
        self.assertEqual(report.samples, 0)
        self.assertGreater(report.requested_samples, 1)
        self.assertIsNone(report.q_lower)
        self.assertIsNone(report.q_upper)
        self.assertIsNone(report.residual_bound)

    @patch("gaussian_synthesis.validation.log_second_moment_bound",
           return_value=math.log(.01))
    @patch("gaussian_synthesis.validation.batch_moments", side_effect=fixed_moments)
    def test_joint_euler_interval_uses_one_confidence_ball(self, moment, bound):
        sampler = FixedSampler()
        report = validate_extent_candidate(
            self.psi, self.psi, 2, sampler, CountingBudget(),
            delta=.25, margin=.1, batch_size=2, max_samples=32)
        self.assertEqual(report.status, "certified")
        self.assertEqual(report.samples, report.requested_samples)
        self.assertEqual(sampler.draws, report.samples)
        radius = math.sqrt(.01 / (report.samples*.25))
        self.assertAlmostEqual(report.statistical_radius, radius)
        self.assertAlmostEqual(report.q_estimate, .875)
        self.assertAlmostEqual(report.q_lower, .875-radius)
        self.assertAlmostEqual(report.q_upper, .875+radius)
        self.assertAlmostEqual(report.residual_estimate, .125)
        self.assertAlmostEqual(report.residual_bound, .125+radius)
        self.assertEqual(report.delta, .25)
        json.dumps(report.summary(), allow_nan=False)

    @patch("gaussian_synthesis.validation.log_second_moment_bound",
           return_value=math.log(.01))
    @patch("gaussian_synthesis.validation.batch_moments", side_effect=fixed_moments)
    def test_interrupted_partial_samples_have_diagnostics_not_certificate(self, moment, bound):
        sampler = FixedSampler()
        report = validate_extent_candidate(
            self.psi, self.psi, 2, sampler, CountingBudget(stop_at=3),
            delta=.25, margin=.1, batch_size=2, max_samples=32)
        self.assertEqual(report.status, "resource_limit")
        self.assertEqual(report.samples, 2)
        self.assertGreater(report.requested_samples, report.samples)
        self.assertAlmostEqual(report.q_estimate, .875)
        self.assertAlmostEqual(report.residual_estimate, .125)
        self.assertIsNotNone(report.reconstruction)
        self.assertIsNone(report.q_lower)
        self.assertIsNone(report.q_upper)
        self.assertIsNone(report.statistical_radius)
        self.assertIsNone(report.residual_bound)

    @patch("gaussian_synthesis.validation.log_second_moment_bound",
           return_value=math.log(.01))
    def test_inconsistent_intersection_is_not_certified(self, bound):
        def impossible_moment(u, k, samples, *, n):
            return MomentEstimate(2., 2*u, 0., 0., len(samples))
        with patch("gaussian_synthesis.validation.batch_moments",
                   side_effect=impossible_moment):
            report = validate_extent_candidate(
                self.psi, self.psi, 2, FixedSampler(), CountingBudget(),
                delta=.25, margin=.1, batch_size=2, max_samples=32)
        self.assertEqual(report.status, "inconsistent_bounds")
        self.assertAlmostEqual(report.q_estimate, 2.)
        self.assertIsNone(report.q_lower)
        self.assertIsNone(report.q_upper)
        self.assertIsNone(report.residual_bound)

    def test_invalid_precision_resources_and_inputs(self):
        options = [
            {"delta": value} for value in (0., 1., math.nan, math.inf, True)
        ] + [
            {"margin": value} for value in (0., math.nan, math.inf, True)
        ] + [
            {"batch_size": value} for value in (0, 1.5, 8193, True)
        ] + [
            {"max_samples": value} for value in (0, 1.5, 12_000_001, True)
        ]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_extent_candidate(
                    self.psi, self.psi, 2, NoDrawSampler(), CountingBudget(), **kwargs)
        for k in (0, False, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                validate_extent_candidate(
                    self.psi, self.psi, k, NoDrawSampler(), CountingBudget())
        bad = self.psi.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            validate_extent_candidate(
                self.psi, bad, 2, NoDrawSampler(), CountingBudget())

    def test_float_overflow_is_controlled_before_sampling(self):
        report = validate_extent_candidate(
            self.psi, 1e100*self.psi, 4, NoDrawSampler(), CountingBudget())
        self.assertEqual(report.status, "numerical_error")
        self.assertEqual(report.samples, 0)
        self.assertIsNone(report.q_upper)
        self.assertIsNotNone(report.error)
        json.dumps(report.summary(), allow_nan=False)
