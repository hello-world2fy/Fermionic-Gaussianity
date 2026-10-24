import unittest
import numpy as np
from fk4.rounding import corrected_cost
from fk.model import build_model
from fk3.discovery import covariance_from_kets


class GaussianRoundingTests(unittest.TestCase):
    def test_constructed_gaussian_and_fock_repair_obey_bound(self):
        model=build_model();rng=np.random.default_rng(81293)
        masks=model.even_masks
        for noise in [0.,1e-4,.03]:
            v=np.zeros(128,complex);v[0]=1
            r=rng.normal(size=128)+1j*rng.normal(size=128);r/=np.linalg.norm(r)
            v+=noise*r;v/=np.linalg.norm(v)
            result=corrected_cost(v[None],np.array([1.+0j]),v,model.j_matrices)
            cov=covariance_from_kets(v)[0]
            H=np.zeros((256,256),complex)
            for a in range(16):
                for b in range(a+1,16):H-=1j*cov[a,b]*(model.gammas[a]@model.gammas[b])
            _,vectors=np.linalg.eigh(H[np.ix_(masks,masks)])
            g=vectors[:,0];overlap=np.vdot(g,v);g*=overlap/abs(overlap)
            gaussian_cov=covariance_from_kets(g)[0]
            self.assertLess(np.max(abs(gaussian_cov@gaussian_cov+np.eye(16))),2e-12)
            self.assertLessEqual(np.linalg.norm(v-g),result['per_atom_exact_Gaussian_distance_upper'])
            actual_cost=1+np.sum(abs(v-g))
            self.assertLessEqual(actual_cost,result['corrected_Gaussian_cost_upper'])

    def test_far_non_gaussian_state_does_not_receive_nearby_certificate(self):
        model=build_model();v=model.S[:,0]
        with self.assertRaises(ArithmeticError):
            corrected_cost(v[None],np.array([1.+0j]),v,model.j_matrices)
