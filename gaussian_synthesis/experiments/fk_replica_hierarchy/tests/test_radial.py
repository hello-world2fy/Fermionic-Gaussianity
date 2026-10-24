import unittest
import numpy as np
from pathlib import Path
from fk4 import paths
from fk4.radial import exact_radial_scale,target_overlap
from fk4.cartan_sparse import CartanMoments
from fk.model import build_model
from fk2.exact_moments import ExactK2Moments,full_tensor_reconstruction


class RadialTests(unittest.TestCase):
    def test_exact_k2_full_tensor_and_k4_gradient_elimination(self):
        model=build_model();target=model.target(11);z_target=model.S.conj().T@target
        rng=np.random.default_rng(61211)
        backends={2:ExactK2Moments(model)}
        path=Path(__file__).resolve().parents[1]/'results/cartan_exact_k4_01/moment_matrix.npz'
        if path.exists():backends[4]=CartanMoments(4,np.load(path)['matrix'])
        for k,backend in backends.items():
            for _ in range(8):
                z=z_target+.2*rng.normal(size=3)
                v=model.S@z
                value=backend.evaluate(z,hessian=False)
                scale=exact_radial_scale(v,target,value.q,k)
                scaled=backend.evaluate(z*scale,hessian=False)
                lam=target_overlap(v,target)
                np.testing.assert_allclose(scaled.gradient,lam/value.q*value.gradient,atol=2e-12,rtol=2e-11)
                if k==2:
                    expected=lam/value.q*full_tensor_reconstruction(v)
                    np.testing.assert_allclose(full_tensor_reconstruction(v*scale),expected,atol=1e-12)
                for r in (.2,4.):
                    value_r=backend.evaluate(z*r,hessian=False)
                    scale_r=exact_radial_scale(v*r,target,value_r.q,k)
                    np.testing.assert_allclose(scale_r*r,scale,rtol=1e-12)


if __name__=='__main__':unittest.main()
