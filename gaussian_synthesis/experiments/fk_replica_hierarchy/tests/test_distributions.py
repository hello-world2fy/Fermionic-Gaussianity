import unittest
import numpy as np
from fk4 import paths
from fk4.distributions import full_distribution
from fk.model import build_model
from fk.gaussian import sample_gaussian
from fk.observables import atom_invariants


class DistributionTests(unittest.TestCase):
    def test_all_particle_features_match_independently_sampled_covariances(self):
        model = build_model()
        sample = sample_gaussian(8, 40, np.random.default_rng(4561))
        a, r = full_distribution(sample.kets, model.j_matrices)
        independent = atom_invariants(sample.covariances, model.j_matrices)
        for i, name in enumerate(("mass_a", "mass_b", "mass_dot", "eta")):
            np.testing.assert_allclose(a["features"][:, i], independent[name], atol=1e-13)
        self.assertLess(r["max_purity_error_all_particles"], 1e-12)
        self.assertEqual(a["mass_a_eta_counts"].sum(), 40)


if __name__ == "__main__":
    unittest.main()
