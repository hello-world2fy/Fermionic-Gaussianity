"""Small structural tests; these do not run the formal follow-up experiment."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(EXPERIMENT.parent / "fk_two_cluster"))

from fk.dictionaries import Dictionary
from fk.model import build_model
from fk.runtime import Budget
from fk2.rb import Candidate, PairedAccumulator, hopping_gaussian_probe, run_rb


class RaoBlackwellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def test_direct_paired_means_and_variances(self):
        model = self.model
        dictionary = Dictionary("full")
        psi = model.target(11.)
        candidate = Candidate("test", 11., 2, psi.copy(), psi.copy(), "test", "unused")
        batch = dictionary.sample(32, np.random.default_rng(911))
        accumulator = PairedAccumulator(dictionary, model.S, [candidate])
        accumulator.add(batch.kets[:13])
        accumulator.add(batch.kets[13:])
        overlap = np.einsum("ni,i->n", batch.kets.conj(), psi, optimize=False)
        coefficients = np.exp(dictionary.log_frame_constant(2))*abs(overlap)**2*overlap
        raw_atoms = coefficients[:, None]*batch.kets
        projected_atoms = np.einsum("ij,ni->nj", model.S.conj(), raw_atoms, optimize=False)
        orbit_atoms = np.einsum("ij,nj->ni", model.S, projected_atoms, optimize=False)
        row = accumulator.row(0, .01)
        raw_mean, orbit_mean, _ = accumulator.means()
        np.testing.assert_allclose(raw_mean[0], raw_atoms.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(orbit_mean[0], orbit_atoms.mean(axis=0), atol=1e-12)
        raw_variance = np.sum(np.abs(raw_atoms-raw_atoms.mean(axis=0))**2)/31
        orbit_variance = np.sum(np.abs(orbit_atoms-orbit_atoms.mean(axis=0))**2)/31
        self.assertAlmostEqual(row["raw_sample_variance_trace"], raw_variance, places=10)
        self.assertAlmostEqual(row["orbit_sample_variance_trace"], orbit_variance, places=10)
        self.assertGreaterEqual(row["paired_variance_ratio"], 1-1e-12)
        self.assertLess(row["paired_pythagoras_error"], 1e-12)
        self.assertLess(row["projection_commutation_error"], 1e-12)
        self.assertFalse(row["formally_certified"])
        self.assertFalse(row["original_first_round_acceptance_changed"])

    def test_hopping_probe_is_invariant_gaussian(self):
        probe, metadata = hopping_gaussian_probe(self.model)
        self.assertAlmostEqual(metadata["hopping_ground_energy"], -4., places=12)
        self.assertLess(metadata["gaussian_purity_error"], 1e-11)
        self.assertLess(metadata["invariance_error"], 1e-11)
        self.assertAlmostEqual(float(np.linalg.norm(probe)), 1., places=12)

    def _source(self, directory, name):
        psi = self.model.target(11.)
        key = "n65536_t11_k2"
        digest = hashlib.sha256(psi.tobytes()).hexdigest()
        document = {"rows": [{"dictionary": name, "seed": 7, "t": 11, "k": 2,
                              "train_samples": 65536, "key": key,
                              "optimizer_converged": True, "witness_sha256": digest}]}
        (directory / f"pilot_{name}_7.json").write_text(json.dumps(document))
        np.savez_compressed(directory / f"witnesses_{name}_7.npz", **{key+"_u": psi})

    def test_small_run_source_freeze_and_extra_candidate(self):
        with tempfile.TemporaryDirectory(prefix="fk_rb_test_") as tmp:
            source = Path(tmp)
            self._source(source, "full")
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            config = {"dictionary": "full", "seed": 7, "t_values": [11], "orders": [2],
                      "sample_sizes": [16, 32], "batch_size": 8, "rng_purpose": 801}
            probe, _ = hopping_gaussian_probe(self.model)
            extra = [{"label": "exact_control", "t": 11, "k": 2, "u": probe,
                      "expected_reconstruction": probe, "expected_q": 1.}]
            checkpoints = []
            result, arrays = run_rb(self.model, source, config, Budget(),
                                    lambda report, saved: checkpoints.append(report["completed_samples"]), extra)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(checkpoints, [16, 32])
            self.assertEqual(len(result["rows"]), 6)
            self.assertEqual({r["sourcekind"] for r in result["rows"]},
                             {"frozen_pilot", "hopping_gaussian_probe", "deterministic_k2"})
            self.assertEqual(arrays["n32_raw_C"].shape, (3, 128))
            self.assertEqual(arrays["n32_orbit_C"].shape, (3, 128))
            self.assertEqual(result["rng_seed_sequence"], [7, 0, 801])
            probe_row = next(row for row in result["rows"] if row["sourcekind"] == "hopping_gaussian_probe")
            expected_check = (abs(probe_row["q_estimate"]-1) <= 5*probe_row["q_standard_error_exact"]+1e-12
                              and abs(probe_row["mass_estimate"]-probe_row["expected_mass"])
                              <= 5*probe_row["mass_standard_error_exact"]+1e-12
                              and probe_row["raw_full_residual"] <= 6*probe_row["raw_rmse_exact"]+1e-12)
            self.assertEqual(probe_row["probe_diagnostic_calibration_passed"], expected_check)
            self.assertFalse(result["formal_confidence_guarantee"])
            frozen = next(row for row in result["rows"] if row["sourcekind"] == "frozen_pilot")
            self.assertEqual(frozen["exact_reference_kind"], "deterministic_full_k2_cartan_projector")
            self.assertGreater(frozen["exact_bias_l2"], 0)
            self.assertIsNotNone(frozen["raw_error_to_exact_C"])
            self.assertAlmostEqual(frozen["raw_full_residual"]**2,
                                   frozen["raw_error_to_exact_C"]**2 + frozen["exact_bias_l2"]**2
                                   + frozen["raw_bias_noise_cross_term"], places=12)
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            self.assertEqual(before, after)

    def test_product_does_not_mislabel_hopping_probe(self):
        with tempfile.TemporaryDirectory(prefix="fk_rb_product_test_") as tmp:
            source = Path(tmp)
            self._source(source, "product")
            config = {"dictionary": "product", "seed": 7, "t_values": [11], "orders": [2],
                      "sample_sizes": [16], "batch_size": 8, "rng_purpose": 802}
            result, _ = run_rb(self.model, source, config, Budget())
            self.assertEqual(result["candidate_count"], 1)
            self.assertIsNone(result["probe"])
            self.assertEqual(result["rows"][0]["sourcekind"], "frozen_pilot")
            config["rng_purpose"] = 2
            with self.assertRaises(ValueError):
                run_rb(self.model, source, config, Budget())


if __name__ == "__main__":
    unittest.main()
