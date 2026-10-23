"""Structural and invariant-measure checks for both FK dictionaries."""

import math
import unittest
import numpy as np
from fk.gaussian import sample_gaussian
from fk.dictionaries import Dictionary


def majoranas(n):
    """Independent dense reference in the complete little-endian Fock basis."""
    result = np.zeros((2*n, 1 << n, 1 << n), dtype=complex)
    for mode in range(n):
        for source in range(1 << n):
            destination = source ^ (1 << mode)
            sign = (-1)**((source & ((1 << mode)-1)).bit_count())
            result[2*mode, destination, source] = sign
            result[2*mode+1, destination, source] = 1j*sign*(1-2*((source >> mode) & 1))
    return result


def covariance_from_ket(ket, n, parity):
    masks = [m for m in range(1 << n) if m.bit_count() % 2 == parity]
    full = np.zeros(1 << n, dtype=complex)
    full[masks] = ket
    # <g|gamma_a gamma_b|g> = <gamma_a g|gamma_b g>.
    acted = majoranas(n) @ full
    covariance = (1j*(acted.conj() @ acted.T)).real
    np.fill_diagonal(covariance, 0.)
    return covariance


class GaussianSamplerTests(unittest.TestCase):
    def test_ket_covariance_and_purity(self):
        for n in (2, 4, 8):
            for parity in (0, 1):
                batch = sample_gaussian(n, 3, np.random.default_rng(104+n+parity), parity)
                np.testing.assert_allclose(np.sum(abs(batch.kets)**2, axis=1), 1, atol=2e-13)
                np.testing.assert_allclose(batch.covariances + batch.covariances.transpose(0, 2, 1), 0, atol=2e-14)
                np.testing.assert_allclose(batch.covariances @ batch.covariances,
                                           np.broadcast_to(-np.eye(2*n), (3, 2*n, 2*n)), atol=2e-13)
                for ket, covariance in zip(batch.kets, batch.covariances):
                    np.testing.assert_allclose(covariance_from_ket(ket, n, parity), covariance, atol=3e-13)

    def test_full_frame_and_probe_moments(self):
        dictionary = Dictionary("full")
        sample = dictionary.sample(8192, np.random.default_rng(7301))
        # A fixed vacuum is a Gaussian probe.  The identity
        # E|<0|g>|^(2k) = 1/d_(8,k) checks more than normalization.
        overlap_squared = abs(sample.kets[:, 0])**2
        for k in (1, 2, 3):
            values = np.exp(dictionary.log_frame_constant(k))*overlap_squared**k
            self.assertLess(abs(values.mean()-1), 7*values.std(ddof=1)/math.sqrt(len(values)))
        frame = sample.kets.T @ sample.kets.conj()/len(sample.kets)
        # Mean squared Frobenius frame error is (1-1/D)/N.
        self.assertLess(np.linalg.norm(frame-np.eye(128)/128), 2/math.sqrt(len(sample.kets)))

    def test_product_parities_frame_and_zero_cross_covariance(self):
        dictionary = Dictionary("product")
        sample = dictionary.sample(8192, np.random.default_rng(7302))
        self.assertGreater(np.count_nonzero(sample.sectors == 0), 3500)
        self.assertGreater(np.count_nonzero(sample.sectors == 1), 3500)
        masks = np.array([m for m in range(256) if m.bit_count() % 2 == 0])
        local_parity = np.array([(int(m) & 15).bit_count() % 2 for m in masks])
        self.assertEqual(np.max(abs(sample.kets[sample.sectors == 0][:, local_parity == 1])), 0)
        self.assertEqual(np.max(abs(sample.kets[sample.sectors == 1][:, local_parity == 0])), 0)
        self.assertEqual(np.max(abs(sample.covariances[:, :8, 8:])), 0)
        np.testing.assert_allclose(np.sum(abs(sample.kets)**2, axis=1), 1, atol=2e-13)
        for parity in (0, 1):
            row = np.flatnonzero(sample.sectors == parity)[0]
            np.testing.assert_allclose(covariance_from_ket(sample.kets[row], 8, 0),
                                       sample.covariances[row], atol=3e-13)
        frame = sample.kets.T @ sample.kets.conj()/len(sample.kets)
        self.assertLess(np.linalg.norm(frame-np.eye(128)/128), 2/math.sqrt(len(sample.kets)))
        # Vacuum and |1_A,1_B> probe the two orthogonal local parity blocks.
        for index in (0, int(np.flatnonzero(masks == 17)[0])):
            for k in (1, 2):
                values = np.exp(dictionary.log_frame_constant(k))*abs(sample.kets[:, index])**(2*k)
                self.assertLess(abs(values.mean()-1), 7*values.std(ddof=1)/math.sqrt(len(values)))

    def test_analytic_frame_constants_and_invalid_inputs(self):
        full, product = Dictionary("full"), Dictionary("product")
        self.assertAlmostEqual(math.exp(full.log_frame_constant(1)), 128, places=10)
        self.assertAlmostEqual(math.exp(product.log_frame_constant(1)), 128, places=10)
        self.assertAlmostEqual(math.exp(product.log_frame_constant(2)), 2*35**2, places=9)
        self.assertNotAlmostEqual(full.log_frame_constant(2), product.log_frame_constant(2))
        with self.assertRaises(ValueError):
            Dictionary("even_even_only")
        with self.assertRaises(ValueError):
            full.log_frame_constant(0)
        with self.assertRaises(ValueError):
            full.sample(0, np.random.default_rng(1))

    def test_model_wick_energy_mass_and_schmidt_bounds(self):
        from fk.model import build_model
        from fk.observables import (atom_invariants, covariance,
                                    gaussian_energy_from_covariance,
                                    mass_correlation_wick)
        model = build_model()
        for name in ("full", "product"):
            batch = Dictionary(name).sample(12, np.random.default_rng(5100 + (name == "product")))
            states = np.zeros((len(batch.kets), 256), dtype=complex)
            states[:, model.even_masks] = batch.kets
            for t in (0., 9., 20.):
                explicit_energy = np.einsum("bi,ij,bj->b", states.conj(),
                                             model.full_hamiltonian(t), states, optimize=False)
                wick_energy = gaussian_energy_from_covariance(batch.covariances, t, model.j_matrices)
                np.testing.assert_allclose(explicit_energy.imag, 0, atol=3e-12)
                np.testing.assert_allclose(explicit_energy.real, wick_energy, atol=3e-12)
            explicit_cq = np.einsum("bi,ij,bj->b", states.conj(),
                                      model.mass_correlation_operator, states, optimize=False)
            np.testing.assert_allclose(explicit_cq.imag, 0, atol=3e-12)
            np.testing.assert_allclose(explicit_cq.real,
                                       mass_correlation_wick(batch.covariances, model.j_matrices),
                                       atol=3e-12)
            invariants = atom_invariants(batch.covariances, model.j_matrices)
            self.assertTrue(np.all(invariants["mass_a"] + invariants["eta"] <= 1 + 1e-12))
            self.assertTrue(np.all(invariants["mass_b"] + invariants["eta"] <= 1 + 1e-12))
            for row, state in enumerate(states):
                schmidt_l1 = np.linalg.svd(state.reshape(16, 16), compute_uv=False).sum()
                bound = (1 + np.sqrt(invariants["eta"][row]))**2
                self.assertLessEqual(schmidt_l1, bound + 1e-12)
                np.testing.assert_allclose(covariance(batch.kets[row], model),
                                           batch.covariances[row], atol=3e-13)


if __name__ == "__main__":
    unittest.main()
