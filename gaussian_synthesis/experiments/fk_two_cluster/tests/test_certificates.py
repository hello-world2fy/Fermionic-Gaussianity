"""Small algebraic checks; no Haar sampler or FK model is needed."""
import unittest
import numpy as np
from fk.baselines import project_atoms, solve_projected_atomic_l1, target_error_estimate
from fk.certificates import (necessary_structure_bounds, repaired_cost,
                             restricted_lower_bound, separation_certificate)


class CertificateTests(unittest.TestCase):
    def test_fock_repair_adds_absolute_residual(self):
        residual = np.array([0.3j, -0.4])
        self.assertAlmostEqual(repaired_cost(1.2, residual), 1.9)
        self.assertAlmostEqual(repaired_cost(1.2, residual, target_error=0.1),
                               1.9 + np.sqrt(2) * 0.1)

    def test_uncertainty_weakens_both_bounds(self):
        r = np.zeros(128)
        exact = separation_certificate(1.2, r, 3.8852976173)
        uncertain = separation_certificate(1.2, r, 3.8852976173, state_error=0.01)
        self.assertGreater(uncertain["cost_upper"], exact["cost_upper"])
        self.assertLess(uncertain["restricted_cost_lower"], exact["restricted_cost_lower"])
        self.assertAlmostEqual(restricted_lower_bound(3.0, 0.25, state_error=0.1), 2.6 / 2.25)
        self.assertFalse(exact["strict_interval_certified"])
        self.assertFalse(exact["statistically_certified"])
        self.assertLess(uncertain["mean_eta_necessary_lower"], exact["mean_eta_necessary_lower"])
        self.assertLess(uncertain["eta_above_threshold_mass_fraction_lower"],
                        exact["eta_above_threshold_mass_fraction_lower"])

    def test_structure_necessity_formulae(self):
        result = necessary_structure_bounds(3.8852976173, 1.2)
        ratio = 3.8852976173 / 1.2
        self.assertAlmostEqual(result["mean_eta_necessary_lower"], (np.sqrt(ratio)-1)**2)
        self.assertAlmostEqual(result["eta_above_threshold_mass_fraction_lower"],
                               (ratio-2.25)/1.75)
        small_ratio = necessary_structure_bounds(1.0, 2.0)
        self.assertEqual(small_ratio["mean_eta_necessary_lower"], 0.0)
        self.assertEqual(small_ratio["eta_above_threshold_mass_fraction_lower"], 0.0)
        boundary = necessary_structure_bounds(4.0, 1.0, eta0=1.0)
        self.assertEqual(boundary["eta_above_threshold_mass_fraction_lower"], 0.0)

    def test_target_residual_gap_estimate_has_explicit_limitations(self):
        class DiagonalModel:
            even_masks = np.arange(128)

            def full_hamiltonian(self, t):
                return np.diag(np.arange(256, dtype=float)*2.0)

        angle = 1e-3
        psi = np.zeros(128)
        psi[:2] = (np.cos(angle), np.sin(angle))
        result = target_error_estimate(DiagonalModel(), 0, psi)
        phase_aligned_error = np.linalg.norm(psi-np.eye(128)[0])
        self.assertGreater(result["target_state_error_estimate"], phase_aligned_error)
        self.assertAlmostEqual(result["target_full_spectral_gap_estimate"], 2.0)
        self.assertFalse(result["target_state_error_interval_certified"])
        self.assertIn("floating-point", result["target_state_error_interpretation"])

    def test_projected_atoms_are_not_normalized(self):
        S = np.array([[1.0], [0.0]])
        atoms = np.array([[0.5, np.sqrt(0.75)]])
        self.assertAlmostEqual(abs(project_atoms(S, atoms)[0, 0]), 0.5)
        result = solve_projected_atomic_l1(S, atoms, np.array([1.0, 0.0]))
        self.assertAlmostEqual(result["coefficient_l1"], 2.0, places=7)
        self.assertLess(np.linalg.norm(result["residual"]), 1e-8)

    def test_full_space_residual_not_just_projection(self):
        S = np.array([[1.0], [0.0]])
        atoms = np.array([[1.0, 0.0]])
        result = solve_projected_atomic_l1(S, atoms, np.array([1.0, 0.3j]))
        self.assertAlmostEqual(result["cost_upper"], 1.3, places=7)
        self.assertAlmostEqual(abs(result["residual"][1]), 0.3, places=7)

    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            restricted_lower_bound(4, 1.1)
        with self.assertRaises(ValueError):
            repaired_cost(1, np.zeros(2), target_error=-0.1)
        with self.assertRaises(ValueError):
            separation_certificate(1, np.zeros(2), 4, numerical_margin=-0.1)


if __name__ == "__main__":
    unittest.main()
