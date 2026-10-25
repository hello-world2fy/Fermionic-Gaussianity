import sys
from pathlib import Path
import unittest
from itertools import combinations
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.gaussian import haar_charts, pfaffian_batch, sample_atoms
from pairing_formation.model import picket_fence
from pairing_formation.replica2 import Replica2


class GaussianTests(unittest.TestCase):
    def test_full_fock_normalization_and_haar_frame(self):
        rng = np.random.default_rng(101)
        n = 4
        Z, vacuum = haar_charts(n, 8192, rng)
        columns = []
        for number in range(0, n+1, 2):
            for indices in combinations(range(n), number):
                value = np.ones(len(Z)) if not indices else pfaffian_batch(Z[:, indices, :][:, :, indices])
                columns.append(vacuum*value)
        atoms = np.array(columns).T
        np.testing.assert_allclose(np.sum(abs(atoms)**2, axis=1), 1., atol=2e-11)
        frame = atoms.conj().T@atoms/len(atoms)
        self.assertLess(np.linalg.norm(frame-np.eye(8)/8), .02)

    def test_fourth_moment_matches_independent_haar(self):
        from math import comb
        model = picket_fence(4, 2.)
        target = model.evolve([.9])[0]
        rng = np.random.default_rng(102)
        values = []
        for _ in range(64):
            atoms, _, _ = sample_atoms(model, 256, rng)
            values.extend(.5*comb(4*model.L, 2*model.L)*abs(atoms.conj()@target)**4)
        values = np.array(values)
        exact = Replica2(model).evaluate(target)[0]
        self.assertLess(abs(values.mean()-exact), 6*values.std()/np.sqrt(len(values)))


if __name__ == '__main__':
    unittest.main()
