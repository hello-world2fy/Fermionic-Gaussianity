"""Convention-sensitive checks for the independent FK reference implementation."""
import unittest

import numpy as np

from fk.conventions import clifford, majoranas, matmul, parity_masks
from fk.exact import run_exact
from fk.model import build_model
from fk.observables import atom_invariants, covariance, gaussian_energy_from_covariance
from fk.symmetry import symmetry_report


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def test_majorana_conventions(self):
        gammas = majoranas(4)
        identity = np.eye(16)
        for a in range(8):
            for b in range(8):
                np.testing.assert_allclose(matmul(gammas[a],gammas[b])+matmul(gammas[b],gammas[a]),
                                           2*(a==b)*identity, atol=1e-14)
        vacuum = np.zeros(128, complex)
        vacuum[0] = 1
        cov = covariance(vacuum, self.model)
        np.testing.assert_allclose(cov[np.arange(0,16,2),np.arange(1,16,2)], -1., atol=1e-14)
        self.assertEqual(majoranas(2)[2,2,0], 1)
        np.testing.assert_array_equal(self.model.even_masks, parity_masks(8,0))

    def test_clifford(self):
        matrices = clifford()
        np.testing.assert_array_equal(matrices+matrices.transpose(0,2,1), 0)
        for a in range(7):
            for b in range(7):
                np.testing.assert_array_equal(matmul(matrices[a],matrices[b])+matmul(matrices[b],matrices[a]),
                                              -2*(a==b)*np.eye(8))

    def test_complete_singlet_space(self):
        report = symmetry_report(self.model)
        self.assertEqual(report["fixed_space_dimension"], 3)
        self.assertTrue(report["verified"])

    def test_exact_reference_and_general_wick(self):
        report = run_exact(self.model, [0,9,20])
        self.assertTrue(report["structure"]["verified"])
        zero, critical, twenty = report["points"]
        self.assertAlmostEqual(zero["energy"], -56, places=10)
        self.assertAlmostEqual(zero["schmidt_l1"], 1, places=10)
        self.assertAlmostEqual(critical["gap"], 18, places=9)
        self.assertAlmostEqual(twenty["schmidt_l1"], 3.8852976173158393, places=10)
        self.assertAlmostEqual(twenty["wick_deviation"], -2.61462609366684, places=10)
        self.assertAlmostEqual(report["wick_peak"]["t_over_v"], 10.9224676, places=5)
        self.assertAlmostEqual(report["wick_peak"]["absolute_value"], 5.02015136569, places=9)

    def test_mean_field_saturating_covariances(self):
        j = clifford()
        for t in [0,3,9,20]:
            s = min(t/9,1.)
            a = np.sqrt(1-s*s)
            cov = np.block([[a*j[0],-s*np.eye(8)],[s*np.eye(8),-a*j[0]]])
            np.testing.assert_allclose(matmul(cov,cov), -np.eye(16), atol=1e-13)
            invariants = atom_invariants(cov,j)
            self.assertAlmostEqual(float(invariants["mass_a"]+invariants["eta"]), 1, places=12)
            expected = -32-2*t*t/9 if t <= 9 else -14-4*t
            self.assertAlmostEqual(float(gaussian_energy_from_covariance(cov,t,j)), expected, places=11)


if __name__ == "__main__":
    unittest.main()
