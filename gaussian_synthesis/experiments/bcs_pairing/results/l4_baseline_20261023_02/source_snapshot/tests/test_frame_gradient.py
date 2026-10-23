import unittest
import numpy as np
from gaussian_synthesis import FockBasis, GaussianState, GaussianCoefficientMeasure
from gaussian_synthesis.moments import batch_moments
from gaussian_synthesis.objective import ConvexObjective
from gaussian_synthesis.sampling import HaarFourModeSampler
from gaussian_synthesis.solver import project_ball
from gaussian_synthesis.theory import dimension


class FrameGradientTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(73)
        self.g = HaarFourModeSampler(self.rng).sample(512)
        self.u = (self.rng.normal(size=8)+1j*self.rng.normal(size=8))/4
        self.psi = (self.rng.normal(size=8)+1j*self.rng.normal(size=8))/4

    def test_frame_dimension_and_k1(self):
        for n in range(2, 9):
            self.assertEqual(dimension(n, 1), 2**(n-1))
        value, grad = ConvexObjective(self.psi, 1).value_gradient(self.u)
        np.testing.assert_allclose(grad, self.u-self.psi)
        self.assertAlmostEqual(value, np.vdot(self.u, self.u).real/2-np.vdot(self.u, self.psi).real)
        sampler = HaarFourModeSampler(self.rng)
        frame = np.zeros((8, 8), complex)
        for _ in range(4):
            g = sampler.sample(8192)
            frame += g.T@g.conj()/32768
        self.assertLess(np.linalg.norm(frame-np.eye(8)/8), .012)

    def test_real_directional_gradient(self):
        v = self.rng.normal(size=8)+1j*self.rng.normal(size=8)
        for k in (2, 4):
            objective = ConvexObjective(self.psi, k)
            _, grad = objective.value_gradient(self.u, self.g)
            h = 1e-6
            fd = (objective.value_gradient(self.u+h*v, self.g)[0]
                  -objective.value_gradient(self.u-h*v, self.g)[0])/(2*h)
            self.assertAlmostEqual(fd, np.vdot(grad, v).real, places=7)

    def test_phase_invariance_and_euler_identity(self):
        phase = np.exp(1j*self.rng.uniform(-np.pi, np.pi, len(self.g)))
        for k in (1, 2, 4):
            a = batch_moments(self.u, k, self.g)
            b = batch_moments(self.u, k, phase[:, None]*self.g)
            np.testing.assert_allclose(a.C, b.C, atol=1e-13)
            self.assertAlmostEqual(a.q, b.q, places=12)
            self.assertAlmostEqual(np.vdot(self.u, a.C).real, a.q, places=12)

    def test_measure_density_reconstructs_same_C(self):
        measure = GaussianCoefficientMeasure(FockBasis(), 2, self.u)
        reconstructed = np.zeros(8, complex)
        for row in self.g[:32]:
            state = GaussianState(FockBasis(), row)
            reconstructed += measure.density(state)*measure.aligned_state(state).amplitudes/32
        np.testing.assert_allclose(reconstructed, batch_moments(self.u, 2, self.g[:32]).C, atol=1e-12)

    def test_zero_overlap_and_projection(self):
        measure = GaussianCoefficientMeasure(FockBasis(), 2, FockBasis().ket(0))
        atom = GaussianState(FockBasis(), FockBasis().ket(15))
        self.assertEqual(measure.density(atom), 0)
        np.testing.assert_equal(measure.aligned_state(atom).amplitudes, atom.amplitudes)
        z = batch_moments(np.zeros(8), 1, self.g)
        self.assertEqual(z.q, 0)
        np.testing.assert_equal(z.C, np.zeros(8))
        self.assertAlmostEqual(np.linalg.norm(project_ball(10*self.u, 1)), 1)
