import unittest
import numpy as np
from gaussian_synthesis import FockBasis, GaussianState
from gaussian_synthesis.gaussian import pfaffian, quadric_matrix, spin_to_fock
from gaussian_synthesis.sampling import HaarFourModeSampler


class BasisGaussianTests(unittest.TestCase):
    def setUp(self):
        self.basis = FockBasis()
        self.rng = np.random.default_rng(70)

    def test_occupation_and_parity(self):
        self.assertEqual(self.basis.occupations, (0, 3, 5, 6, 9, 10, 12, 15))
        self.assertEqual(FockBasis(4, "odd").dimension, 8)
        with self.assertRaises(ValueError):
            self.basis.index(1)
        with self.assertRaises(NotImplementedError):
            HaarFourModeSampler(self.rng, FockBasis(3))

    def test_invalid_and_zero_inputs(self):
        for bad in (np.zeros(7), np.full(8, np.nan), np.zeros(8)):
            with self.assertRaises(ValueError):
                self.basis.vector(bad, nonzero=True)

    def test_pfaffian_signs_and_determinant(self):
        self.assertEqual(pfaffian(np.empty((0, 0))), 1)
        self.assertEqual(pfaffian(np.zeros((3, 3))), 0)
        a = self.rng.normal(size=(4, 4))+1j*self.rng.normal(size=(4, 4))
        z = a-a.T
        expected = z[0, 1]*z[2, 3]-z[0, 2]*z[1, 3]+z[0, 3]*z[1, 2]
        self.assertAlmostEqual(abs(pfaffian(z)-expected), 0, places=11)
        a = self.rng.normal(size=(6, 6))+1j*self.rng.normal(size=(6, 6))
        z = a-a.T
        np.testing.assert_allclose(pfaffian(z)**2, np.linalg.det(z), rtol=1e-11)

    def test_pairing_amplitudes(self):
        a = self.rng.normal(size=(4, 4))+1j*self.rng.normal(size=(4, 4))
        z = a-a.T
        g = GaussianState.from_pairing(z)
        self.assertAlmostEqual(np.linalg.norm(g.to_fock()), 1, places=12)
        np.testing.assert_allclose(g.amplitudes[1]/g.amplitudes[0], z[0, 1])
        np.testing.assert_allclose(g.amplitudes[-1]/g.amplitudes[0],
            z[0, 1]*z[2, 3]-z[0, 2]*z[1, 3]+z[0, 3]*z[1, 2])

    def test_zero_vacuum_chart_and_nongaussian_rejection(self):
        g = GaussianState(self.basis, self.basis.ket(15))
        self.assertEqual(g.amplitudes[0], 0)
        with self.assertRaises(ValueError):
            GaussianState(self.basis, (self.basis.ket(0)+self.basis.ket(15))/np.sqrt(2))

    def test_spin_mapping_and_sample_normalization(self):
        U = spin_to_fock(np.eye(8)).T
        np.testing.assert_allclose(U.conj().T@U, np.eye(8), atol=1e-14)
        np.testing.assert_allclose(U.T@quadric_matrix()@U, np.eye(8), atol=1e-14)
        g = HaarFourModeSampler(self.rng).sample(256)
        np.testing.assert_allclose(np.linalg.norm(g, axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(np.einsum("bi,ij,bj->b", g, quadric_matrix(), g), 0, atol=1e-14)
        with self.assertRaises(ValueError):
            HaarFourModeSampler(self.rng).sample(8193)
