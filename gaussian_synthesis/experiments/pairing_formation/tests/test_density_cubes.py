import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.jastrow import JastrowDynamics
from pairing_formation.density_cubes import mixed_swap_cubes, cube_diagnostics, disjoint_cube_bound, EVEN, ODD


class DensityCubeTests(unittest.TestCase):
    def test_quadratic_radial_identity_and_bound(self):
        model = picket_fence(8, 2.)
        cubes = mixed_swap_cubes(model)
        rng = np.random.default_rng(291)
        # Independent raw polynomial construction, including nontrivial radial
        # magnitudes and phases, rather than using the quotient feature basis.
        for _ in range(4):
            J = rng.normal(0, .3, (model.L, model.L))+1j*rng.normal(0, .3, (model.L, model.L))
            radial = rng.normal(size=model.r+1)+1j*rng.normal(size=model.r+1)
            raw = np.einsum('si,ij,sj->s', model.occ, J, model.occ)+radial[model.m]
            state = np.exp(raw-np.max(raw.real)); state /= np.linalg.norm(state)
            mass, invariant, gap = cube_diagnostics(abs(state)**2, cubes)
            self.assertLess(np.max(abs(invariant)), 2e-16)
            target = rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
            target /= np.linalg.norm(target)
            bound = disjoint_cube_bound(abs(target)**2, cubes)
            self.assertLessEqual(abs(np.vdot(target, state))**2, bound['fidelity_upper_bound_float64']+1e-14)
        # Count-dependent orbitals can violate the quadratic identity: two
        # interband bits couple to the third (intraband energy-choice) bit.
        raw = .8*(model.m-2)**2*(model.occ[:, 0]-model.occ[:, 1])
        p = np.exp(raw-np.max(raw)); p /= p.sum()
        self.assertGreater(disjoint_cube_bound(p, cubes)['infidelity_lower_bound_float64'], 1e-7)


if __name__ == '__main__':
    unittest.main()
