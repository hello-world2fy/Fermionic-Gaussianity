import unittest
import numpy as np
from fk4 import paths
from fk4.energy_windows import hopping_eigenbasis, arc_projection_x, window_construction, x_cap, restricted_support_upper
from fk4.columns import independent_arc_dictionary
from fk.model import build_model
from fk.gaussian import sample_gaussian
from fk.observables import gaussian_energy_from_covariance


class EnergyWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def test_arc_projection_against_independent_quadratic_ground_states(self):
        m = self.model
        V = hopping_eigenbasis()
        np.testing.assert_allclose(V.T@m.reduced_hopping@V, np.diag([-4, 0, 4]), atol=1e-13)
        atoms, angles = independent_arc_dictionary(m, 25)
        projection = atoms@m.S.conj()@V
        projection *= (projection[:, 0].conj()/abs(projection[:, 0]))[:, None]
        for p, theta in zip(projection, angles):
            np.testing.assert_allclose(p, arc_projection_x(1-np.sin(theta)), atol=1e-12)

    def test_full_gaussian_energy_and_opposite_vacuum_bounds(self):
        m = self.model
        sample = sample_gaussian(8, 128, np.random.default_rng(8901))
        h = m.h_hop[np.ix_(m.even_masks, m.even_masks)]
        x = 1+np.einsum('ni,ij,nj->n', sample.kets.conj(), h, sample.kets).real/4
        opposite = m.S@hopping_eigenbasis()[:, 2]
        self.assertTrue(np.all(abs(sample.kets.conj()@opposite) <= x*x/4+1e-12))
        for t in (9, 9.1, 11, 20):
            energy = gaussian_energy_from_covariance(sample.covariances, t, m.j_matrices)
            self.assertTrue(np.all(energy-(-14-4*t) >= 18*x*x+4*(t-9)*x-1e-11))

    def test_constructive_bounds_at_small_windows_without_fock_repair(self):
        m = self.model
        for t in (9, 9.1, 11, 20):
            z = m.S.conj().T@m.target(t)
            for delta in (1e-5, 1e-3, .1):
                for allowance in (0, 1e-3):
                    r = window_construction(z, t, delta, error_allowance=allowance)
                    self.assertLess(r["high_precision_reconstruction_error"], 1e-60)
                    self.assertLessEqual(r["lower_cost"], r["upper_cost"])
                    self.assertLess(r["maximum_component_energy_excess"], delta)
                    self.assertLessEqual(r["approximation_error_used"], allowance+1e-15)
                    self.assertFalse(r["uses_out_of_window_fock_repair"])


if __name__ == "__main__":
    unittest.main()
