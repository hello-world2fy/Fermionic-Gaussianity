"""Checks for canonical evidence gates, independent reconstructions, and streams."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT.parent/"fk_two_cluster", ROOT.parent/"fk_two_cluster_followup"):
    sys.path.insert(0, str(entry))

import numpy as np

from fk.dictionaries import Dictionary
from fk.model import build_model
from fk.moments import ProjectedMoments, pack
from fk.runtime import ResourceLimit
from fk2.exact_moments import full_tensor_reconstruction
from fk3.canonical import prepare_canonical, train_k4


class ExhaustedBudget:
    def check(self):
        raise ResourceLimit("test_budget_exhausted")

    def usage(self):
        return {"seconds": 0.}


class CanonicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def test_integer_and_full_tensor_audits_gate_physical_targets(self):
        report, arrays = prepare_canonical(self.model, {"t_values": [0, 5, 9, 11, 20]}, None)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["audit_passed"])
        self.assertTrue(report["verified"])
        self.assertFalse(report["strict_interval_certified"])
        self.assertEqual(arrays["matrix"].shape, (6, 6))
        np.testing.assert_array_equal(arrays["integer_gram"], arrays["integer_gram_expected"])
        self.assertEqual(len(report["rows"]), 5)
        for row in report["rows"]:
            u = arrays[row["witness_key"]]
            target = arrays[row["target_key"]]
            self.assertLess(np.linalg.norm(full_tensor_reconstruction(u)-target), 1e-8)
            self.assertTrue(row["reconstruction_accepted"])
            self.assertIsNone(row["coefficient_mass"])
        self.assertEqual(len(report["gaussian_probes"]), 5)
        self.assertGreater(report["gaussian_exact_reference"]["mass"], 1.)

    def test_incompatible_singlet_phase_blocks_closed_form_reuse(self):
        changed = self.model.S.copy()
        changed[:, 0] *= 1j
        bad_model = replace(self.model, S=changed)
        report, _ = prepare_canonical(bad_model, {"t_values": [0]}, None)
        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "audit_failed")
        self.assertFalse(report["audit_passed"])
        self.assertEqual(report["rows"], [])
        self.assertGreater(report["audit"]["basis_error"], 1.)

    def test_budget_failure_preserves_nonaccepted_partial_state(self):
        saved = []
        report, arrays = prepare_canonical(self.model, {}, ExhaustedBudget(),
                                          lambda r, a: saved.append((r["status"], list(a))))
        self.assertEqual(report["status"], "resource_limit")
        self.assertFalse(report["verified"])
        self.assertEqual(report["rows"], [])
        self.assertIn("S", arrays)
        self.assertEqual(saved[-1][0], "resource_limit")
        training, cache = train_k4(self.model, {"sample_sizes": [64]}, 20261023, ExhaustedBudget())
        self.assertEqual(training["status"], "resource_limit")
        self.assertEqual(cache["h_cache"].shape, (0, 3))
        self.assertEqual(int(cache["sample_count"]), 0)

    def test_k4_prefix_snapshots_replay_and_do_not_claim_validation(self):
        config = {"sample_sizes": [64, 128], "k4_t_values": [0], "batch_size": 64}
        report, arrays = train_k4(self.model, config, 20261023, None)
        self.assertEqual(report["status"], "trained")
        self.assertFalse(report["heldout_validation_completed"])
        self.assertEqual(report["rng_seed_sequence"], [20261023, 910, 1])
        self.assertEqual(arrays["h_cache"].shape, (128, 3))
        rng = np.random.default_rng(np.random.SeedSequence([20261023, 910, 1]))
        batch = Dictionary("full").sample(64, rng)
        expected = np.einsum("ni,ij->nj", batch.kets.conj(), self.model.S, optimize=False)
        np.testing.assert_allclose(arrays["h_cache"][:64], expected, atol=1e-15, rtol=0)
        for row in report["rows"]:
            n = row["train_samples"]
            backend = ProjectedMoments(arrays["h_cache"][:n], 4, Dictionary("full").log_frame_constant(4))
            z = arrays[f"z_t0_k4_n{n}"]
            target = self.model.S.conj().T@arrays["target_t0"]
            value = backend.evaluate(z, hessian=False)
            self.assertAlmostEqual(value.q, row["q_train"], places=12)
            self.assertLess(np.linalg.norm(value.gradient-pack(target)), 1e-10)
            self.assertFalse(row["reconstruction_accepted"])
            self.assertFalse(row["heldout_validation_completed"])


if __name__ == "__main__":
    unittest.main()
