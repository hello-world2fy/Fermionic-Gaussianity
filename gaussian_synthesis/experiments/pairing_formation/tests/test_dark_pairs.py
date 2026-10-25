import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model, annihilator
from pairing_formation.dark_pairs import PairDarkSectors
from pairing_formation.frozen import channel_basis


class DarkPairTests(unittest.TestCase):
    def test_full_physical_annihilators(self):
        model, forms = angular_model(10)
        dark = PairDarkSectors(model, forms)
        rng = np.random.default_rng(941)
        state = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
        particle = dark.apply(state, 'particle')
        hole = dark.apply(state, 'hole')
        for f in forms.T:
            lower_upper = annihilator(model.L, model.r, np.r_[np.zeros(model.r), f])
            fill_lower = annihilator(model.L, model.r + 1, np.r_[f, np.zeros(model.r)]).conj().T
            self.assertLess(np.linalg.norm(lower_upper @ particle), 2e-13)
            self.assertLess(np.linalg.norm(fill_lower @ hole), 2e-13)
        self.assertGreater(np.linalg.norm(particle), .1)
        self.assertLess(np.linalg.norm(dark.apply(model.initial, 'either')), 1e-14)

    def test_projectors_and_probabilities(self):
        model, forms = angular_model(10)
        dark = PairDarkSectors(model, forms)
        rng = np.random.default_rng(942)
        states = rng.normal(size=(model.dim, 3)) + 1j * rng.normal(size=(model.dim, 3))
        for kind in ('both', 'either', 'particle', 'hole'):
            projected = dark.apply(states, kind)
            np.testing.assert_allclose(dark.apply(projected, kind), projected, atol=2e-14)
            for j in range(3):
                np.testing.assert_allclose(dark.apply(states[:, j], kind), projected[:, j], atol=2e-14)
                key = {'both': 'both_dark', 'either': 'either_dark',
                       'particle': 'particle_dark', 'hole': 'hole_dark'}[kind]
                self.assertAlmostEqual(sum(dark.evaluate(states[:, j])[key]),
                                       float(np.linalg.norm(projected[:, j])**2), places=11)

    def test_source_gaussian_span_is_bright(self):
        model, forms = angular_model(10)
        U, _ = np.linalg.qr(forms)
        basis, _ = channel_basis(model, U @ U.T, 2)
        dark = PairDarkSectors(model, forms)
        self.assertLess(np.linalg.norm(dark.apply(basis, 'either')), 2e-12)


if __name__ == '__main__':
    unittest.main()
