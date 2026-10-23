"""Independent algebra checks for the streaming structure estimators."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent / "fk_two_cluster", ROOT.parent / "fk_two_cluster_followup"):
    sys.path.insert(0, str(path))

from fk.model import build_model
from fk.dictionaries import Dictionary
from fk.observables import atom_invariants
from fk.runtime import Budget, ResourceLimit
from fk3.statistics import (StructureAccumulator, cross_kernel, eta_groups,
                            invariant_bins, run_statistics, _prepare_candidates)


class StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def _candidate(self):
        target = self.model.target(11.)
        return {"id": "test_k2", "t": 11., "k": 2, "u": target.copy(),
                "source": "test", "target": target.copy(), "deterministic_verified": False}

    def test_bin_and_group_boundaries(self):
        eta = np.asarray([0., .25, .5, .75, 1.])
        np.testing.assert_array_equal(eta_groups(eta), [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(invariant_bins(eta, eta), [0, 5, 10, 15, 15])

    def test_streamed_sums_match_direct_calculation(self):
        prepared, _ = _prepare_candidates(self.model, [self._candidate()], False)
        accumulator = StructureAccumulator(self.model, prepared)
        batch = Dictionary("full").sample(37, np.random.default_rng(971))
        accumulator.add(batch.kets[:13], batch.covariances[:13])
        accumulator.add(batch.kets[13:], batch.covariances[13:])
        rows, means = accumulator.rows({"planned_statistical_events": 5, "phase": "final"})
        kets, u = batch.kets, prepared[0]["u"]
        overlap = kets.conj() @ u
        coefficients = np.exp(Dictionary("full").log_frame_constant(2))*abs(overlap)**2*overlap
        mass = abs(coefficients)
        coords = kets @ self.model.S.conj()
        inv = atom_invariants(batch.covariances, self.model.j_matrices)
        groups = eta_groups(inv["eta"])
        bins = invariant_bins(inv["eta"], inv["mass_a"])
        raw = np.mean(coefficients[:, None]*kets, axis=0)
        projected = np.mean(coefficients[:, None]*coords, axis=0)
        np.testing.assert_allclose(means["raw_C"][0], raw, atol=2e-13)
        np.testing.assert_allclose(means["rb_C"][0], self.model.S @ projected, atol=2e-13)
        np.testing.assert_allclose(means["histogram"][0].ravel(),
            np.bincount(bins, weights=mass, minlength=16)/mass.sum(), atol=2e-14)
        expected_half = np.zeros((2, 3, 3), complex)
        for h in range(2):
            half = np.arange(37) % 2 == h
            for group in range(3):
                chosen = half & (groups == group)
                expected_half[h, group] = np.sum(coefficients[chosen, None]*coords[chosen], axis=0)/sum(half)
        np.testing.assert_allclose(means["group_phi_halves"][0], expected_half, atol=2e-13)
        np.testing.assert_allclose(means["K_cross"][0],
            cross_kernel(expected_half[0], expected_half[1], accumulator.operator), atol=2e-12)
        self.assertLess(rows[0]["group_additivity_error"], 1e-12)
        self.assertLess(rows[0]["pythagoras_error"], 1e-11)
        self.assertLess(rows[0]["projection_commutation_error"], 1e-12)
        self.assertAlmostEqual(rows[0]["mean_eta"], np.dot(mass, inv["eta"])/mass.sum(), places=12)
        for group in range(3):
            atoms = coefficients[:, None]*coords*(groups == group)[:, None]
            sample_var = np.sum(abs(atoms-atoms.mean(axis=0))**2)/(37-1)
            self.assertAlmostEqual(rows[0]["group_rmse"][group], np.sqrt(sample_var/37), places=12)
        self.assertFalse(rows[0]["histogram_formally_certified"])
        self.assertFalse(rows[0]["group_rmse_is_confidence_radius"])
        self.assertEqual(rows[0]["formal_envelopes"]["allocated_events_this_candidate"], 5)

    def test_cross_kernel_avoids_self_noise_bias(self):
        operator = np.diag([1., 2., 3.])
        left = np.asarray([[1., 0, 0], [0, 2., 0]], complex)
        right = np.asarray([[2., 0, 0], [0, 1., 0]], complex)
        kernel = cross_kernel(left, right, operator)
        expected = (left.conj() @ operator @ right.T+right.conj() @ operator @ left.T)/2
        np.testing.assert_allclose(kernel, expected)
        np.testing.assert_allclose(kernel, kernel.conj().T)
        average = (left+right)/2
        self.assertGreater(np.linalg.norm(kernel-average.conj() @ operator @ average.T), .1)

    def test_phases_are_independent_and_preserve_candidate(self):
        candidate = self._candidate()
        original = candidate["u"].copy()
        config = {"sample_sizes": [16, 32], "final_samples": 32, "batch_size": 8,
                  "planned_statistical_events": 5, "include_probes": False}
        snapshots = []
        dev, arrays = run_statistics(self.model, [candidate], config, 111, Budget(),
            lambda report, arrays: snapshots.append(report["completed_samples"]), phase="development")
        final, final_arrays = run_statistics(self.model, [candidate], config, 111, Budget(), phase="final")
        self.assertEqual(dev["status"], "complete")
        self.assertEqual(final["status"], "complete")
        self.assertEqual(dev["rng_seed_sequence"], [111, 910, 2])
        self.assertEqual(final["rng_seed_sequence"], [111, 910, 4])
        self.assertEqual(snapshots, [16, 32, 32])
        np.testing.assert_array_equal(candidate["u"], original)
        self.assertGreater(np.linalg.norm(arrays["n32_raw_C"]-final_arrays["n32_raw_C"]), 1e-6)
        expected_tv = .5*np.sum(abs(arrays["n32_histogram"]-arrays["n16_histogram"]))
        self.assertAlmostEqual(dev["rows"][-1]["histogram_tv_previous"], expected_tv)
        self.assertIsNone(final["rows"][0]["histogram_tv_previous"])
        self.assertFalse(dev["rows"][0]["formal_envelopes"]["included_in_joint_budget"])
        self.assertTrue(final["rows"][0]["formal_envelopes"]["included_in_joint_budget"])

    def test_probe_events_not_claimed_and_caps_enforced(self):
        config = {"sample_sizes": [16], "final_samples": 16, "batch_size": 8,
                  "planned_statistical_events": 5}
        report, arrays = run_statistics(self.model, [self._candidate()], config, 222, Budget())
        probes = [row for row in report["rows"] if row["is_probe"]]
        self.assertEqual(len(probes), 2)
        for probe in probes:
            self.assertEqual(probe["formal_envelopes"]["allocated_events_this_candidate"], 0)
            self.assertFalse(probe["formal_envelopes"]["included_in_joint_budget"])
            self.assertIn("probe_passed", probe)
        self.assertEqual(arrays["n16_histogram"].shape, (3, 4, 4))
        with self.assertRaises(ValueError):
            run_statistics(self.model, [self._candidate()], {"sample_sizes": [65537]}, 222, Budget())

    def test_partial_batch_alarm_rolls_back_consistent_state(self):
        original_add = StructureAccumulator.add
        calls = []
        def interrupted_add(accumulator, kets, covariance):
            calls.append(1)
            if len(calls) == 2:
                accumulator.raw_sum += 12345
                raise ResourceLimit("synthetic_mid_batch_alarm")
            return original_add(accumulator, kets, covariance)
        config = {"sample_sizes": [8, 16], "batch_size": 8, "include_probes": False}
        with patch.object(StructureAccumulator, "add", interrupted_add):
            report, arrays = run_statistics(self.model, [self._candidate()], config, 333, Budget())
        self.assertEqual(report["status"], "resource_limit")
        self.assertEqual(report["completed_samples"], 8)
        self.assertEqual(len(report["rows"]), 1)
        np.testing.assert_allclose(arrays["raw_sum"]/8, arrays["n8_raw_C"])
        self.assertEqual(int(arrays["completed_samples"]), 8)
        self.assertNotIn("n16_raw_C", arrays)


if __name__ == "__main__":
    unittest.main()
