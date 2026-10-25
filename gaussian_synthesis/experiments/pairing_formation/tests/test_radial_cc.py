import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.coupled_cluster import objective_T, fit_cc
from pairing_formation.fitting import fit_product_family


class RadialCCTests(unittest.TestCase):
    def test_radial_reverse_derivative_and_rank1_agreement(self):
        model = picket_fence(8, 2.)
        target = model.evolve([1.4])[0]
        rng = np.random.default_rng(993)
        T = .3*(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
        direction = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
        loss, gradient, state = objective_T(model, target, T, radial=True)
        eps = 2e-6
        finite = (objective_T(model, target, T+eps*direction, radial=True)[0]
                  -objective_T(model, target, T-eps*direction, radial=True)[0])/(2*eps)
        self.assertAlmostEqual(finite, 2*np.vdot(gradient, direction).real, places=8)
        self.assertAlmostEqual(-loss, abs(np.vdot(target, state))**2, places=13)
        radial, _ = fit_product_family(model, target, radial=True, starts=3)
        cc, _, _ = fit_cc(model, target, 1, radial=True, starts=3)
        self.assertAlmostEqual(cc['best_found_fidelity'], radial['fidelity'], places=9)


if __name__ == '__main__':
    unittest.main()
