import unittest
import numpy as np
from fk4 import paths
from fk4.geometry import contact_hessian
from fk4.tempering import actions
from fk4.state_enclosure import enclose_ground_state
from fk.gaussian import sample_gaussian
from fk.model import build_model


class GeometryAndStateTests(unittest.TestCase):
    def test_gaussian_reference_has_56_nondegenerate_normal_directions(self):
        g=sample_gaussian(8,1,np.random.default_rng(28102)).kets[0]
        report,raw=contact_hessian(g,g)
        self.assertLess(report['horizontal_gradient_norm'],1e-12)
        np.testing.assert_allclose(raw['hessian_eigenvalues'],2.,atol=1e-12)

    def test_generator_hessian_against_direct_gaussian_rotation(self):
        model=build_model()
        g=sample_gaussian(8,1,np.random.default_rng(28103)).kets[0]
        w=model.target(11)
        _,raw=contact_hessian(w,g)
        permutations,phases=actions(8)
        step=1e-4
        f=lambda ket:-np.log(abs(np.vdot(w,ket))**2)
        for j in [0,17,73,119]:
            acted=phases[j]*g[permutations[j]]
            numerical=(f(np.cos(step)*g+np.sin(step)*acted)-2*f(g)+f(np.cos(step)*g-np.sin(step)*acted))/step**2
            self.assertAlmostEqual(numerical,raw['generator_hessian'][j,j],delta=3e-6)

    def test_full_spectrum_enclosure_including_intentional_target_error(self):
        model=build_model()
        t=20
        target=model.target(t)
        exact=enclose_ground_state(model,t,target)
        self.assertLess(exact['saved_target_to_exact_ground_state_distance_upper'],1e-12)
        H=model.full_hamiltonian(t)[np.ix_(model.even_masks,model.even_masks)]
        _,V=np.linalg.eigh(H)
        perturb=V[:,1]
        damaged=target+1e-4*perturb
        damaged/=np.linalg.norm(damaged)
        result=enclose_ground_state(model,t,damaged)
        self.assertGreater(result['saved_target_to_exact_ground_state_distance_upper'],1e-4-1e-10)
        self.assertLess(result['saved_target_to_exact_ground_state_distance_upper'],.01)
        self.assertEqual(result['verified_eigenvalues_below_separator'],1)


if __name__=='__main__':
    unittest.main()
