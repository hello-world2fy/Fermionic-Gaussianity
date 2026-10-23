import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gaussian_synthesis import ExtentConfig, FockBasis, estimate_extent
from gaussian_synthesis.solver import ResourceBudget, SolveResult
from gaussian_synthesis.validation import ExtentValidationReport
from _magic_reference import magic_state, magic_q, magic_optimizer


class ExtentApiTests(unittest.TestCase):
    def setUp(self):
        self.basis = FockBasis()
        self.psi = magic_state(self.basis)

    def config(self, **options):
        values = dict(iterations=2, training_batch=16, diagnostic_samples=16,
                      max_seconds=10, max_memory_mib=256)
        values.update(options)
        return ExtentConfig(**values)

    def test_k1_exact_interval_and_serialization(self):
        result = estimate_extent(self.psi, self.basis, self.config(orders=(1,)))
        self.assertTrue(result.interval_certified)
        self.assertFalse(result.target_reached)
        self.assertEqual(result.stop_reason, "orders_exhausted")
        self.assertAlmostEqual(result.lower_bound, 1.)
        self.assertAlmostEqual(result.upper_bound, 2.)
        self.assertEqual(result.confidence, 1.)
        self.assertEqual(result.stages[0].validation.samples, 0)
        with tempfile.TemporaryDirectory() as directory:
            result.save(directory)
            saved = json.loads((Path(directory)/"summary.json").read_text())
            self.assertFalse(saved["target_reached"])
            for name in ("vectors.npz", "history.csv", "intervals.csv"):
                self.assertTrue((Path(directory)/name).is_file())
            with np.load(Path(directory)/"vectors.npz", allow_pickle=False) as vectors:
                np.testing.assert_array_equal(vectors["psi"], self.psi)
                np.testing.assert_array_equal(vectors["u_k1"], self.psi)
            with self.assertRaises(FileExistsError):
                result.save(directory)

    def test_deterministic_fock_bound_stops_before_any_training(self):
        psi = 2j*self.basis.ket(3)
        with patch("gaussian_synthesis.extent.projected_sgd") as solver:
            result = estimate_extent(psi, config=self.config())
        solver.assert_not_called()
        self.assertTrue(result.target_reached)
        self.assertEqual(result.lower_bound, 4.)
        self.assertEqual(result.upper_bound, 4.)
        self.assertEqual(result.confidence, 1.)
        self.assertEqual(result.stages, [])
        np.testing.assert_array_equal(result.psi, psi)

    def test_sample_cap_stops_once_and_retains_previous_interval(self):
        result = estimate_extent(self.psi, config=self.config(max_validation_samples=1))
        self.assertEqual([s.k for s in result.stages], [1, 2])
        stage = result.stages[-1]
        self.assertEqual(stage.validation.status, "sample_budget_exceeded")
        self.assertEqual(stage.validation.samples, 0)
        self.assertIsNone(stage.certificate)
        self.assertFalse(stage.applied)
        self.assertEqual(result.stop_reason, "sample_budget_exceeded")
        self.assertTrue(result.interval_certified)
        self.assertEqual(result.confidence, 1.)
        self.assertFalse(result.target_reached)
        self.assertAlmostEqual(result.lower_bound, 1.)
        self.assertAlmostEqual(result.upper_bound, 2.)

    def test_joint_budget_and_optional_stopping(self):
        calls = []

        def solve(psi, k, sampler, diagnostic_sampler, budget, **kwargs):
            return SolveResult(magic_optimizer(k, self.basis), k, "completed", 2, 2., [])

        def validate(psi, u, k, sampler, budget, **kwargs):
            calls.append((k, kwargs["delta"]))
            q = np.linalg.norm(u)**(2*k)*magic_q(k)
            return ExtentValidationReport(
                "certified", 0, 0, None if k == 1 else kwargs["delta"], "fixture",
                q_estimate=q, q_lower=q, q_upper=q, reconstruction=psi.copy(),
                residual_estimate=0., statistical_radius=0., residual_bound=0.)

        # The exact k=2 certificate reaches 90% width but k=1 does not.
        # Confidence must retain the entire .05 allocation despite stopping
        # before the planned k=4 validation, not shrink to the observed .025.
        with patch("gaussian_synthesis.extent.projected_sgd", side_effect=solve), \
             patch("gaussian_synthesis.extent.validate_extent_candidate", side_effect=validate):
            result = estimate_extent(self.psi, config=self.config(relative_tolerance=.9))
        self.assertEqual(calls, [(1, .025), (2, .025)])
        self.assertTrue(result.target_reached)
        self.assertEqual(result.confidence, .95)
        self.assertAlmostEqual(result.lower_bound, magic_q(2)**(-.5))
        self.assertAlmostEqual(result.upper_bound, 2.)
        self.assertTrue(all(s.applied for s in result.stages))

    def test_general_complex_input_reproducibility_and_independent_streams(self):
        rng = np.random.default_rng(92)
        psi = rng.normal(size=8)+1j*rng.normal(size=8)
        psi /= np.linalg.norm(psi)
        config = self.config(orders=(1, 2), validation_margin=1., max_validation_samples=10000)
        first = estimate_extent(psi, config=config)
        second = estimate_extent(psi, config=config)
        self.assertEqual(first.stop_reason, "orders_exhausted")
        self.assertEqual(first.confidence, .95)
        self.assertTrue(first.stages[-1].applied)
        self.assertEqual(first.lower_bound, second.lower_bound)
        self.assertEqual(first.upper_bound, second.upper_bound)
        np.testing.assert_array_equal(first.witnesses[2], second.witnesses[2])
        self.assertTrue(np.all(np.abs(first.witnesses[2][1:7]) > 1e-5))
        keys = [tuple(key) for s in first.stages for key in s.rng_spawn_keys.values()]
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(first.stages[1].validation.delta, .05)
        self.assertEqual(first.stages[1].validation.samples,
                         first.stages[1].validation.requested_samples)
        np.testing.assert_array_equal(first.psi, psi)
        json.dumps(first.summary(), allow_nan=False)

    def test_expired_budget_returns_baseline_without_claiming_target(self):
        expired = ResourceBudget(.001, 256, started=0.)
        result = estimate_extent(self.psi, config=self.config(), budget=expired)
        self.assertEqual(result.stop_reason, "resource_limit")
        self.assertTrue(result.interval_certified)
        self.assertFalse(result.target_reached)
        self.assertEqual(result.stages, [])
        self.assertAlmostEqual(result.upper_bound, 2.)

    def test_invalid_config_input_and_relaxed_budget(self):
        cases = ({"orders": (4, 2)}, {"orders": (1, 8)}, {"orders": (True,)},
                 {"iterations": 8001}, {"training_batch": 8193},
                 {"max_validation_samples": 12_000_001}, {"validation_margin": 0.},
                 {"relative_tolerance": np.nan}, {"max_seconds": 181},
                 {"max_memory_mib": 513}, {"total_failure_probability": np.nan})
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                ExtentConfig(**case)
        for psi in (np.zeros(8), np.ones(16), np.full(8, np.nan)):
            with self.assertRaises(ValueError):
                estimate_extent(psi, config=self.config())
        with self.assertRaises(NotImplementedError):
            estimate_extent(np.ones(4), FockBasis(3), self.config())
        with self.assertRaises(ValueError):
            estimate_extent(self.psi, config=self.config(), budget=ResourceBudget(180, 512))


if __name__ == "__main__":
    unittest.main()
