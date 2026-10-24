import unittest
import numpy as np
from fk4 import paths
from fk4.columns import fit_orbits, independent_arc_dictionary, finite_simplex_expansion
from fk.model import build_model


class ColumnTests(unittest.TestCase):
    def test_dual_sign_feasibility_and_actual_branch_accounting(self):
        model = build_model()
        arc, _ = independent_arc_dictionary(model, 17)
        atoms = np.concatenate((np.eye(128, dtype=complex), arc))
        for t in (0, 20):
            target = model.target(t)
            fit = fit_orbits(model.S, atoms, target)
            self.assertLessEqual(fit["finite_constraint_max"], 1+1e-7)
            self.assertAlmostEqual(fit["restricted_dual_objective"], fit["U_orbit_repaired"], delta=1e-7)
            report, arrays = finite_simplex_expansion(model, atoms, fit["coefficients"], target)
            close = arrays["finite_coefficients"]@arrays["finite_kets"]+arrays["fock_repair"]
            np.testing.assert_allclose(close, target, atol=1e-13)
            cost = abs(arrays["finite_coefficients"]).sum()+abs(arrays["fock_repair"]).sum()
            self.assertAlmostEqual(report["U_finite_repaired"], cost)
            self.assertEqual(report["main_gaussian_branch_count"], len(arrays["finite_kets"]))
            self.assertLess(report["gaussian_purity_error"], 1e-9)


if __name__ == "__main__":
    unittest.main()
