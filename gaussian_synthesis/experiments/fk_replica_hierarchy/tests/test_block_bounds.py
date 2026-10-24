import unittest
from pathlib import Path
import numpy as np
from fk4 import paths
from fk4.cartan_sparse import CartanMoments


class BlockBoundTests(unittest.TestCase):
    def test_exact_low_order_projectors_are_submultiplicative(self):
        root=Path(__file__).resolve().parents[1]
        files={k:root/f'results/cartan_exact_k{k}_01/moment_matrix.npz' for k in [2,3,4]}
        if not all(p.exists() for p in files.values()):self.skipTest('exact-moment regression artifacts absent')
        backends={k:CartanMoments(k,np.load(p)['matrix']) for k,p in files.items()}
        rng=np.random.default_rng(61302)
        for _ in range(64):
            z=rng.normal(size=3)+1j*rng.normal(size=3);z/=np.linalg.norm(z)
            q={1:float(np.vdot(z,z).real),**{k:b.evaluate(z,hessian=False).q for k,b in backends.items()}}
            for a,b in [(1,1),(1,2),(1,3),(2,2)]:
                self.assertLessEqual(q[a+b],q[a]*q[b]+2e-13)


if __name__=='__main__':unittest.main()
