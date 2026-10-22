import tempfile
import unittest
from pathlib import Path
import numpy as np
from gaussian_synthesis import FockBasis, SynthesisConfig, synthesize
from gaussian_synthesis.sampling import HaarFourModeSampler
from gaussian_synthesis.solver import ResourceBudget, ResourceLimitError, projected_sgd
from gaussian_synthesis.validation import validate
from _magic_reference import magic_state


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.basis = FockBasis()
        self.psi = magic_state(self.basis)
        self.sampler = HaarFourModeSampler(np.random.default_rng(83))
        self.budget = ResourceBudget(10, 256)

    def test_known_residual_and_sharper_bound(self):
        report = validate(self.psi, self.psi, 2, self.sampler, self.budget,
                          delta=.1, tolerance=.5, margin=.2, max_samples=10000, k1_mass_samples=32)
        # C_2(M)=0.875 M, so the true residual is exactly 0.125.
        self.assertEqual(report.samples, report.requested_samples)
        self.assertLessEqual(.125, report.upper_bound)
        self.assertLessEqual(report.statistical_radius, .2)
        self.assertAlmostEqual(report.original_pdf_radius/report.statistical_radius, np.sqrt(112))

    def test_sample_budget_rejects_before_sampling(self):
        class NoDraws:
            basis = FockBasis()
            def sample(self, count):
                raise AssertionError("Budget failure must not draw samples")
        report = validate(self.psi, 10*self.psi, 4, NoDraws(), self.budget,
                          max_samples=100, k1_mass_samples=1)
        self.assertEqual(report.status, "sample_budget_exceeded")
        self.assertEqual(report.samples, 0)
        self.assertIsNone(report.upper_bound)

    def test_exact_k1_and_reproducible_serialization(self):
        config = SynthesisConfig(orders=(1,), k1_mass_samples=64,
                                 max_validation_samples=64, max_seconds=10, max_memory_mib=256)
        first = synthesize(self.psi, self.basis, config)
        second = synthesize(self.psi, self.basis, config)
        self.assertEqual(first.status, "certified")
        self.assertEqual(first.stages[0].validation.upper_bound, 0)
        self.assertEqual(first.stages[0].validation.mass_estimate, second.stages[0].validation.mass_estimate)
        keys = first.stages[0].rng_spawn_keys
        self.assertEqual(len({tuple(key) for key in keys.values()}), 3)
        with tempfile.TemporaryDirectory() as directory:
            first.save(directory)
            self.assertTrue((Path(directory)/"summary.json").is_file())
            with np.load(Path(directory)/"vectors.npz") as saved:
                np.testing.assert_array_equal(saved["u_k1"], self.psi)

    def test_delta_allocation_and_no_hidden_full_run(self):
        config = SynthesisConfig(iterations=2, training_batch=16, diagnostic_samples=16,
                                 k1_mass_samples=16, max_validation_samples=16,
                                 max_seconds=10, max_memory_mib=256)
        result = synthesize(self.psi, self.basis, config)
        self.assertEqual(len(result.stages), 3)
        for stage in result.stages[1:]:
            self.assertEqual(stage.solution.iterations, 2)
            self.assertEqual(stage.validation.delta, .025)
            self.assertEqual(stage.validation.status, "sample_budget_exceeded")
        self.assertEqual(result.status, "not_certified")

    def test_resource_stop_has_no_partial_certificate(self):
        expired = ResourceBudget(.001, 256, started=0)
        with self.assertRaises(ResourceLimitError):
            expired.check()
        report = validate(self.psi, self.psi, 2, self.sampler, expired,
                          delta=.1, tolerance=.5, margin=.2, max_samples=10000, k1_mass_samples=1)
        self.assertEqual(report.status, "resource_limit")
        self.assertIsNone(report.upper_bound)
        stopped = synthesize(self.psi, self.basis, SynthesisConfig(orders=(1,)), budget=expired)
        self.assertEqual(stopped.status, "resource_limit")

    def test_theory_mode_averages_preupdate_iterates(self):
        result = projected_sgd(self.psi, 2, self.sampler,
            HaarFourModeSampler(np.random.default_rng(84)), self.budget,
            mode="theory", iterations=2, batch_size=16, diagnostic_samples=16)
        # First gradient at zero is exactly -psi. For two pre-update iterates,
        # their mean is eta*psi/2, independently of the second sampled batch.
        step = result.history[0]["step_size"]
        np.testing.assert_allclose(result.u, step*self.psi/2, atol=1e-15)

    def test_config_hard_limits(self):
        for options in ({"training_batch": 8193}, {"orders": (1, 8)},
                        {"max_validation_samples": 50_000_000}, {"max_seconds": 181},
                        {"max_memory_mib": 513}):
            with self.assertRaises(ValueError):
                SynthesisConfig(**options)
        with self.assertRaises(ValueError):
            synthesize(self.psi, self.basis, SynthesisConfig(max_seconds=10),
                       budget=ResourceBudget(180, 512))
