import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence, PairModel
from pairing_formation.sectors import (resolved_observables, sector_components,
                                       conditional_rank_one, phase_align)


class SectorTests(unittest.TestCase):
    def test_flat_band_reduces_to_collective_chain(self):
        # Independent analytic r+1 Hamiltonian: no hidden additional directions.
        model = PairModel(np.r_[np.full(4, -.5), np.full(4, .5)], 2.)
        r, g = model.r, model.g
        m = np.arange(r+1)
        diagonal = -r+2*m-g*(r+2*m*(r-m))
        hopping = -g*(r-m[:-1])*(m[:-1]+1)
        H = np.diag(diagonal)+np.diag(hopping, 1)+np.diag(hopping, -1)
        lift = (model.m[:, None] == m).astype(float)
        lift /= np.linalg.norm(lift, axis=0)
        np.testing.assert_allclose(model.H@lift, lift@H, atol=2e-14)
        from scipy.linalg import expm
        state = model.evolve([1.8])[0]
        prediction = lift@expm(-1j*1.8*H)[:, 0]
        np.testing.assert_allclose(prediction, state, atol=2e-14)

    def test_continuity_and_operator_norm_bound(self):
        model = picket_fence(8, 2.)
        rng = np.random.default_rng(192)
        for state in [model.evolve([.71])[0], rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)]:
            state /= np.linalg.norm(state)
            obs = resolved_observables(model, state)
            self.assertLess(obs['continuity_error'], 1e-12)
            self.assertLess(obs['coherence_bound_excess'], 1e-12)
            eps = 1e-5
            left, right = model.evolve([-eps, eps], state)
            numerical = (model.observe(right)['variance']-model.observe(left)['variance'])/(2*eps)
            self.assertAlmostEqual(numerical, obs['variance_rate'], places=8)

    def test_count_preserving_controls(self):
        model = picket_fence(8, 2.)
        state = model.evolve([1.83])[0]
        reference = model.observe(state)
        for changed in [phase_align(model, state), conditional_rank_one(model, state), state.conj()]:
            np.testing.assert_allclose(model.observe(changed)['count'], reference['count'], atol=2e-14)
        components = sector_components(model, state)
        np.testing.assert_allclose(np.sum(abs(components)**2, axis=1), abs(state)**2, atol=2e-14)
        obs = resolved_observables(model, state)
        reverse = resolved_observables(model, state.conj())
        np.testing.assert_allclose(reverse['current'], -np.array(obs['current']), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
