import unittest
from dataclasses import FrozenInstanceError
import numpy as np
from gaussian_synthesis import (FermionSpace, FockBasis, OccupationSector, NumberSector,
                                fidelity, phase_aligned_distance)
from gaussian_synthesis.metrics import normalized


class SectorTests(unittest.TestCase):
    def test_metadata_does_not_enumerate(self):
        space = FermionSpace(100)
        self.assertEqual(space.ambient_dimension, 2**99)
        self.assertFalse(hasattr(space, 'occupations'))
        tiny = OccupationSector(space, (0, 3))
        self.assertEqual(tiny.coordinate_dimension, 2)
        with self.assertRaises(ValueError):
            tiny.embed([1, 0], max_dimension=128)

    def test_roundtrip_matches_legacy_even_and_odd_order(self):
        for parity in ('even', 'odd'):
            full = FockBasis(4, parity)
            masks = full.occupations[::-2]
            sector = OccupationSector(full.space, masks)
            vector = np.arange(1, len(masks)+1)+1j
            embedded = sector.embed(vector, max_dimension=8)
            expected = sum((value*full.ket(mask) for value, mask in zip(vector, masks)))
            np.testing.assert_array_equal(embedded, expected)
            np.testing.assert_array_equal(sector.restrict(embedded), vector)
            self.assertAlmostEqual(np.linalg.norm(embedded), np.linalg.norm(vector))

    def test_number_sector_keeps_unpaired_configurations(self):
        number = NumberSector(FermionSpace(8), 4)
        pair = OccupationSector(number.space, (15, 51, 60, 195, 204, 240))
        self.assertEqual((number.dimension, pair.dimension, pair.ambient_dimension), (70, 6, 128))
        self.assertIn(85, number.occupations)
        self.assertNotIn(85, pair.occupations)
        x = np.arange(6)*(.1+.2j)
        embedded = pair.embed_into(x, number, max_dimension=70)
        np.testing.assert_allclose(embedded[[number.index(m) for m in pair.occupations]], x)
        with self.assertRaises(ValueError):
            pair.embed_into(x, NumberSector(FermionSpace(8), 2), max_dimension=70)

    def test_capacity_checked_before_number_enumeration(self):
        with self.assertRaises(ValueError):
            NumberSector(FermionSpace(100), 50, max_dimension=10)
        with self.assertRaises(ValueError):
            NumberSector(FermionSpace(4), 3)
        self.assertEqual(NumberSector(FermionSpace(4, 'odd'), 3).dimension, 4)

    def test_invalid_masks_and_vectors(self):
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                FermionSpace(n)
        for masks in ((), (0, 0), (1,), (-1,), (16,), (True,)):
            with self.assertRaises(ValueError):
                OccupationSector(FermionSpace(4), masks)
        sector = OccupationSector(FermionSpace(4), (0, 3))
        for vector in ([1], [np.nan, 0], [0, 0]):
            with self.assertRaises(ValueError):
                sector.vector(vector, nonzero=True)
        with self.assertRaises(ValueError):
            sector.restrict(np.ones(16))
        with self.assertRaises(ValueError):
            sector.embed([1, 0], max_dimension=True)

    def test_restriction_does_not_normalize(self):
        sector = OccupationSector(FermionSpace(4), (0, 3))
        result = sector.restrict(np.ones(8)/np.sqrt(8))
        self.assertAlmostEqual(np.linalg.norm(result), .5)

    def test_number_metadata_cannot_diverge_from_coordinates(self):
        sector = NumberSector(FermionSpace(4), 2)
        for name, value in [('particle_number', 4), ('occupations', (15,)), ('space', FermionSpace(6))]:
            with self.assertRaises(FrozenInstanceError):
                setattr(sector, name, value)
        self.assertEqual(sector.particle_number, 2)
        self.assertTrue(all(mask.bit_count() == 2 for mask in sector.occupations))


class MetricTests(unittest.TestCase):
    def test_phase_scale_and_squared_overlap(self):
        state = np.array([1, 1j])
        self.assertAlmostEqual(fidelity(state, [1, 0]), .5)
        self.assertLess(phase_aligned_distance(state, -7j*state), 1e-14)
        self.assertAlmostEqual(fidelity(state, 1e200*state), 1.)
        np.testing.assert_allclose(normalized([0, 2]), [0, 1])

    def test_invalid_states(self):
        for state in ([], [0, 0], [np.inf], [[1, 0]]):
            with self.assertRaises(ValueError):
                fidelity(state, state)
        with self.assertRaises(ValueError):
            fidelity([1, 0], [1])
