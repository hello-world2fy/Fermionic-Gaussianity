import unittest
import numpy as np
from fractions import Fraction
from pairing_formation.cartan_monotonicity import paired_cartan_floor,numerical_image_obstruction
from pairing_formation.replica2 import Replica2
from pairing_formation.model import picket_fence


class CartanMonotonicityTests(unittest.TestCase):
    def test_exact_fourteen_level_floor(self):
        mu,_=paired_cartan_floor(14)
        self.assertEqual(mu,Fraction(1301961,8388608))

    def test_random_pairs_obey_global_strong_monotonicity(self):
        rng=np.random.default_rng(1561)
        for L in (4,6,8):
            model=picket_fence(L,2.);replica=Replica2(model);mu=float(paired_cartan_floor(L)[0])
            for _ in range(8):
                u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);u/=np.linalg.norm(u)
                v=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);v*=rng.uniform(.1,3)/np.linalg.norm(v)
                Cu=replica.evaluate(u)[1];Cv=replica.evaluate(v)[1];delta=u-v
                integral=(np.linalg.norm(u)**2+np.linalg.norm(v)**2+np.vdot(u,v).real)/3
                lower=mu/2*np.linalg.norm(delta)**2*integral
                self.assertGreaterEqual(np.vdot(delta,Cu-Cv).real+1e-12,lower)
                self.assertGreaterEqual(np.linalg.norm(Cu-Cv)+1e-12,mu/8*np.linalg.norm(u)**2*np.linalg.norm(delta))

    def test_obstruction_bound_against_sampled_nonlinear_image(self):
        rng=np.random.default_rng(1562);model=picket_fence(6,2.);replica=Replica2(model)
        U=np.linalg.qr(rng.normal(size=(model.dim,3))+1j*rng.normal(size=(model.dim,3)))[0]
        u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        target=replica.evaluate(u)[1];u/=np.linalg.norm(target)**(1/3);target=replica.evaluate(u)[1]
        bound=numerical_image_obstruction(6,u,U,np.linalg.norm(target-target))
        self.assertGreater(bound['infidelity_lower_expression'],0)
        for _ in range(20):
            c=rng.normal(size=3)+1j*rng.normal(size=3);state=replica.evaluate(U@c)[1];state/=np.linalg.norm(state)
            self.assertLessEqual(abs(np.vdot(target,state))**2,bound['fidelity_upper_expression']+1e-12)
