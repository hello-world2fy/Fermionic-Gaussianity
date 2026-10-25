import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.low_cost import (
    blank_coefficient_mass, log_blank_coefficient_mass, minimal_k_within_mass,
    canonical_mean_fidelity, all_to_all, spectator_extension, embed_active_state,
    outside_configuration_fraction, product_gaussian_overlap, certified_onsite_overlap,
    quartic_tilt_energies, edge_bunched_energies)
from pairing_formation.interferometer import interferometer_readout, characteristic_delta, orbit_parity_element


class LowCostInterferometerTests(unittest.TestCase):
    def test_blank_mass_matches_audit_table(self):
        expected = {16: 84.44481175, 20: 315.51524585, 24: 1193.39691876, 28: 4552.44764239}
        minima = {16: 615, 20: 979, 24: 1426, 28: 1957}
        for modes, value in expected.items():
            exact = float(blank_coefficient_mass(modes, 2))
            self.assertAlmostEqual(exact, value, places=6)
            self.assertAlmostEqual(np.exp(log_blank_coefficient_mass(modes, 2)), value, places=6)
            self.assertEqual(minimal_k_within_mass(modes, 1.1), minima[modes])
        self.assertAlmostEqual(canonical_mean_fidelity(24, 2), 9.11e-7, delta=2e-9)
        self.assertEqual(blank_coefficient_mass(4, 2), blank_coefficient_mass(4, 2))

    def test_all_to_all_kernel_matches_picket_fence(self):
        picket = picket_fence(6, 2.7)
        kernel = all_to_all(picket.energies, 2.7)
        np.testing.assert_allclose(kernel.H.toarray(), picket.H.toarray(), atol=1e-12)

    def test_spectator_evolution_is_embedded_active_dynamics(self):
        active = all_to_all(quartic_tilt_energies(4), 3.3)
        extended = spectator_extension(active.energies, 3.3)
        time = 0.7
        evolved = active.evolve([time])[0]
        embedded = embed_active_state(active, extended, evolved)
        big = extended.evolve([time], embed_active_state(active, extended, active.initial))[0]
        overlap = np.vdot(embedded, big)
        self.assertAlmostEqual(abs(overlap), 1, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(big - embedded * overlap / abs(overlap))), 0, places=11)

    def test_dicke_balanced_splitter_matches_closed_form(self):
        model = picket_fence(4, 2)
        state = np.ones(model.dim, complex) / np.sqrt(model.dim)
        readout = interferometer_readout(model, state, (0, 1), (2, 3), 0.0)
        self.assertAlmostEqual(readout['T_real'], 1 / 6, places=12)
        self.assertAlmostEqual(readout['T_imag'], 0, places=12)
        np.testing.assert_allclose(readout['delta_counts'], np.array([1, -4, 6, -4, 1]) / 48, atol=1e-12)
        self.assertAlmostEqual(readout['delta_parity'], 1 / 3, places=12)
        moments = readout['delta_moments_1_to_4']
        self.assertTrue(max(abs(value) for value in moments[:3]) < 1e-11)
        self.assertAlmostEqual(moments[3], 0.5, places=12)
        element = orbit_parity_element(model, state, state, (0, 1), (2, 3), 0.0)
        self.assertAlmostEqual(element.real, readout['raw_parity'], places=12)
        self.assertAlmostEqual(element.imag, 0, places=12)
        quadrature = interferometer_readout(model, state, (0, 1), (2, 3), np.pi / 8)
        self.assertAlmostEqual(quadrature['delta_parity'], 0, places=12)
        self.assertAlmostEqual(quadrature['formula_count_error'], 0, places=12)

    def test_random_seniority_zero_formula_and_characteristic_function(self):
        model = picket_fence(6, 3.3)
        rng = np.random.default_rng(20261025)
        state = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
        state /= np.linalg.norm(state)
        chi = np.linspace(-np.pi, np.pi, 7)
        for phi in (0.0, 0.13, np.pi / 8, np.pi / 5):
            readout = interferometer_readout(model, state, (0, 2), (4, 5), phi)
            self.assertLess(readout['formula_count_error'], 1e-10)
            expectation = readout['T_real'] + 1j * readout['T_imag']
            transform = np.exp(1j * chi[:, None] * np.arange(5)[None, :]) @ readout['delta_counts']
            predicted = [characteristic_delta(expectation, phi, value) for value in chi]
            np.testing.assert_allclose(transform, predicted, atol=1e-10)
            self.assertAlmostEqual(readout['sector_weight'], 1, places=12)

    def test_outside_fraction_and_dicke_onsite_overlap(self):
        self.assertAlmostEqual(outside_configuration_fraction(0.8, 1.0, 0.0, 1.0, 0.5), 0.6)
        self.assertIsNone(outside_configuration_fraction(0.8, 1.0, 0.0, 1.0, 1.0))
        state = np.ones(6, complex) / np.sqrt(6)
        overlap = product_gaussian_overlap(state, np.full(4, np.pi / 4), np.zeros(4))
        self.assertAlmostEqual(abs(overlap), np.sqrt(6) / 4, places=12)
        bound = certified_onsite_overlap(state, tolerance=0.03, max_boxes=20000, starts=4, seed=3)
        self.assertGreaterEqual(bound['overlap_upper'] + 1e-9, np.sqrt(6) / 4)
        self.assertLess(bound['overlap_lower'], np.sqrt(6) / 4 + 1e-8)
        self.assertGreater(bound['overlap_lower'], 0.55)
        self.assertTrue(np.all(np.diff(edge_bunched_energies(6)) >= -1e-12))
        self.assertTrue(np.all(np.diff(quartic_tilt_energies(8)) >= -1e-12))


if __name__ == '__main__':
    unittest.main()
