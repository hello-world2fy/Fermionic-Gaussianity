import unittest
import numpy as np
from fk.moments import ProjectedMoments, pack, unpack
from fk.solver import solve


class SolverTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.h = (rng.normal(size=(200,3))+1j*rng.normal(size=(200,3)))/5
        self.z = np.array([.7+.1j,.2-.4j,.3+.2j])
        self.target = np.array([.8,.3j,.1])

    def test_real_gradient_hessian(self):
        for k in (1,2,3,4):
            b = ProjectedMoments(self.h,k,np.log(128))
            r = b.evaluate(self.z)
            x = pack(self.z)
            step = 1e-5
            grad, hes = np.zeros(6), np.zeros((6,6))
            for j in range(6):
                dx = np.eye(6)[j]*step
                rp, rm = b.evaluate(unpack(x+dx)),b.evaluate(unpack(x-dx))
                grad[j] = (rp.q-rm.q)/(4*k*step)
                hes[:,j] = (rp.gradient-rm.gradient)/(2*step)
            np.testing.assert_allclose(r.gradient,grad,rtol=1e-7,atol=1e-8)
            np.testing.assert_allclose(r.hessian,hes,rtol=1e-7,atol=1e-8)

    def test_newton_stationarity_and_descent(self):
        b = ProjectedMoments(self.h,3,np.log(128))
        result = solve(b,self.target)
        self.assertTrue(result['converged'])
        self.assertLess(np.linalg.norm(b.evaluate(result['z']).gradient-pack(self.target)),1e-8)
        objectives = [r['objective'] for r in result['history']]
        self.assertTrue(np.all(np.diff(objectives)<=1e-10))

    def test_k1_uses_exact_frame_not_empirical_fit(self):
        b = ProjectedMoments(self.h,1,np.log(128))
        np.testing.assert_array_equal(solve(b,self.target)['z'], self.target)
