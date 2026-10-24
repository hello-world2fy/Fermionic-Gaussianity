import unittest
import numpy as np
from fk4 import paths
from fk4.window_certificates import restricted_support,necessity
from fk4.energy_windows import hopping_eigenbasis
from fk4.tempering import actions
from fk.model import build_model


class WindowCertificateTests(unittest.TestCase):
    def test_full_gaussian_near_vacuum_overlaps_respect_interval_window_envelope(self):
        model=build_model();V=hopping_eigenbasis();vacuum=model.S@V[:,0]
        perms,phases=actions(8);rng=np.random.default_rng(61301)
        for t in (9.,9.1,11.,20.):
            H=model.full_hamiltonian(t)[np.ix_(model.even_masks,model.even_masks)]
            EG=-14-4*t
            for _ in range(10):
                g=vacuum.copy()
                for p in rng.permutation(120):
                    angle=rng.normal(scale=.01)
                    g=np.cos(angle)*g+np.sin(angle)*phases[p]*g[perms[p]]
                delta=float(np.vdot(g,H@g).real-EG)+1e-10
                self.assertGreater(delta,0)
                for z in (V[:,2],V[:,1],rng.normal(size=3)+1j*rng.normal(size=3)):
                    bound=restricted_support(z,t,delta)['restricted_support_upper']
                    self.assertLessEqual(abs(np.vdot(g,model.S@z)),bound+1e-12)

    def test_nontrivial_fraction_is_conditional_on_declared_cost(self):
        result=necessity(1.9,.5,2.)
        self.assertTrue(result['nontrivial'])
        self.assertGreaterEqual(result['fraction_upper'],.1)
        self.assertLess(result['fraction_upper'],.100000000000001)
        self.assertFalse(necessity(.1,1.2,2.)['nontrivial'])


if __name__=='__main__':unittest.main()
