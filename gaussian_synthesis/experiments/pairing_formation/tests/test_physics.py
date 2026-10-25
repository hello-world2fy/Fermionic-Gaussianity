import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import PairModel, picket_fence
from pairing_formation.linear import matrices, coupled_cluster_state, squeezed_pair_matrix
from pairing_formation.structure import analyze, collective_basis


class PhysicsTests(unittest.TestCase):
    def test_short_time_exact_coefficients(self):
        for r in (2, 3, 4, 6):
            model = picket_fence(2*r, 2.)
            t = .002
            obs = model.observe(model.evolve([t])[0])
            coefficient = (obs['fano']-1)/(model.G*t/2)**2
            self.assertAlmostEqual(coefficient, 2*(1-1/r)**2-1, delta=5e-6)

    def test_sparse_reference_and_current(self):
        model = picket_fence(8, 2.)
        t, dt = .7, 1e-5
        state = model.evolve([t])[0]
        np.testing.assert_allclose(state, expm(-1j*t*model.H.toarray())@model.initial, atol=2e-13)
        obs = model.observe(state)
        around = model.evolve([t-dt, t+dt])
        derivative = (model.observe(around[1])['mean']-model.observe(around[0])['mean'])/(2*dt)
        # i[H,m] = i*g*(A-A^dagger) = -2g Im(A).
        self.assertAlmostEqual(derivative, -2*model.g*model.r*obs['cross_imag'], places=8)

    def test_self_terms_and_single_transfer(self):
        model = picket_fence(8, 2.)
        P, H, B, _ = matrices(model)
        np.testing.assert_allclose(np.diag(P), 2*model.energies[4:]-model.g)
        np.testing.assert_allclose(np.diag(H), -2*model.energies[:4]+model.g)
        t = 1e-4
        T = squeezed_pair_matrix(model, np.array([t]))[0]
        first = -1j*t*B-.5*t*t*(P@B+B@H.T)
        np.testing.assert_allclose(T, first, atol=1e-12)

    def test_collective_degenerate_limit(self):
        model = PairModel(np.r_[-np.ones(4), np.ones(4)], 4.)
        state = model.evolve([1.3])[0]
        result, radial, _ = analyze(model, state)
        self.assertAlmostEqual(result['conditional_rank1_fidelity_upper'], 1., places=12)
        self.assertAlmostEqual(result['single_direction_radial_fidelity'], 1., places=12)

    def test_rank_one_permanents(self):
        import math
        model = picket_fence(8, 2.)
        rng = np.random.default_rng(17)
        p, h = rng.normal(size=(2, 4))+.1j*rng.normal(size=(2, 4))
        state = coupled_cluster_state(model, np.outer(p, h))
        raw = np.zeros(model.dim, complex)
        for m, (sets, indices) in enumerate(model.blocks):
            a = [np.prod(p[list(s)]) for s in sets]
            b = [np.prod(h[list(s)]) for s in sets]
            raw[indices] = math.factorial(m)*np.outer(a, b)
        np.testing.assert_allclose(state, raw/np.linalg.norm(raw), atol=2e-14)


if __name__ == '__main__':
    unittest.main()
