import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.replica2 import Replica2


class ReplicaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = picket_fence(4, 2.)
        cls.replica = Replica2(cls.model)

    def test_gaussian_and_embedded_magic(self):
        m, r = self.model, self.replica
        q, C = r.evaluate(m.initial)
        self.assertAlmostEqual(q, 1., places=13)
        np.testing.assert_allclose(C, m.initial, atol=1e-14)
        other = m.index[int(m.masks[m.fermi_index]) ^ (1 << 0) ^ (1 << 2)]
        magic = m.initial.copy()
        magic[other] = 1
        magic /= np.sqrt(2)
        q, C = r.evaluate(magic)
        self.assertAlmostEqual(q, .875, places=13)
        np.testing.assert_allclose(C, .875*magic, atol=1e-14)

    def test_real_directional_derivative(self):
        rng = np.random.default_rng(23)
        u = rng.normal(size=6)+1j*rng.normal(size=6)
        v = rng.normal(size=6)+1j*rng.normal(size=6)
        q, C = self.replica.evaluate(u)
        h = 1e-5
        derivative = (self.replica.evaluate(u+h*v)[0]-self.replica.evaluate(u-h*v)[0])/(2*h)
        self.assertAlmostEqual(derivative, 4*np.vdot(C, v).real, delta=8e-6)

    def test_full_majorana_projector_independent(self):
        # Explicit full 2-copy Fock space, including all broken-pair states.
        # This check does not use the beta kernel or its compressed sectors.
        model = self.model
        n = 2*model.L
        D = 1 << n
        rng = np.random.default_rng(38)
        u = rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        u /= np.linalg.norm(u)
        full = np.zeros(D, complex)
        for a, mask in enumerate(model.masks):
            physical = sum(3 << (2*j) for j in range(model.L) if mask & (1 << j))
            full[physical] = u[a]
        product = np.outer(full, full)
        actions = []
        masks = np.arange(D)
        for mode in range(n):
            dst = masks ^ (1 << mode)
            sign = (-1.)**np.bitwise_count(masks & ((1 << mode)-1))
            for imaginary in (False, True):
                phase = sign*(1j*(1-2*((masks >> mode)&1)) if imaginary else 1.)
                actions.append((dst, phase))
        q = 0j
        points = 4*n+1
        for angle in 2*np.pi*np.arange(points)/points:
            v = product.copy()
            for dst, phase in actions:
                acted = v[np.ix_(dst, dst)]*phase[dst, None]*phase[None, dst]
                v = np.cos(angle)*v+1j*np.sin(angle)*acted
            q += np.vdot(product, v)/points
        self.assertAlmostEqual(q.real, self.replica.evaluate(u)[0], places=12)
        self.assertLess(abs(q.imag), 1e-12)

    def test_solve_synthesis_equation(self):
        state = self.model.evolve([.9])[0]
        u, report = self.replica.solve(state)
        self.assertLess(report['residual'], 1e-7)

    def test_response_derivative(self):
        rng = np.random.default_rng(207)
        u = rng.normal(size=6)+1j*rng.normal(size=6)
        v = rng.normal(size=6)+1j*rng.normal(size=6)
        h = 1e-6
        finite = (self.replica.evaluate(u+h*v)[1]-self.replica.evaluate(u-h*v)[1])/(2*h)
        np.testing.assert_allclose(finite, self.replica.derivative(u, v), rtol=1e-8, atol=1e-8)
        response, report = self.replica.witness_response(u, v)
        self.assertEqual(report['cg_status'], 0)
        self.assertLess(report['response_residual'], 1e-8)


if __name__ == '__main__':
    unittest.main()
