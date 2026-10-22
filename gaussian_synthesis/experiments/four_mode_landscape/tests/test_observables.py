"""Small deterministic checks of the streaming Monte Carlo reductions."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
import numpy as np

from gaussian_synthesis.sampling import HaarFourModeSampler
from observables import observe, N_EDGES, XY_EDGES, OCCUPATIONS


def seeds():
    return np.random.SeedSequence(91731).spawn(3)


def target_vectors():
    psi = np.zeros((2, 8), dtype=complex)
    psi[0, 0], psi[0, 7] = np.sqrt(.9), np.sqrt(.1)
    psi[1, [0, 1, 6, 7]] = [.5, .5, .5, .5j]
    u = psi.copy()
    u[0, 7] *= 1.3
    u[1, 0] += .1j
    return psi, u


def sample_in_batches(seed, count, batch):
    sampler = HaarFourModeSampler(np.random.Generator(np.random.PCG64(seed)))
    return np.concatenate([sampler.sample(min(batch, count - offset))
                           for offset in range(0, count, batch)])


class ObservablesTests(unittest.TestCase):
    def test_proportional_distributions_and_matrix_properties(self):
        psi, _ = target_vectors()
        result = observe(psi, 1.7 * np.exp(.41j) * psi,
                         seeds=seeds(), samples_per_repeat=173, batch_size=64)
        np.testing.assert_allclose(result["tv"], 0, atol=3e-15)
        for key in ("R", "purity", "hist_n", "hist_xy", "spectrum", "mean_n", "ess"):
            np.testing.assert_allclose(result[f"{key}_p"], result[f"{key}_h"], atol=2e-13)
        np.testing.assert_allclose(result["mass_p"], 1.7**3 * result["mass_h"])
        np.testing.assert_allclose(result["cost_p"], 1.7**6 * result["cost_h"])
        for kind in ("p", "h"):
            matrices = result[f"R_{kind}"]
            np.testing.assert_allclose(matrices, matrices.conj().transpose(0, 2, 1), atol=1e-14)
            np.testing.assert_allclose(np.trace(matrices, axis1=1, axis2=2), 1, atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(matrices).min(), -1e-14)
            self.assertTrue(np.all(result[f"purity_{kind}"] >= 1 / 8))
            self.assertTrue(np.all(result[f"purity_{kind}"] <= 1 + 1e-14))
            self.assertTrue(np.all(result[f"ess_{kind}"] <= 173 + 1e-10))
            np.testing.assert_allclose(result[f"hist_n_{kind}"].sum(axis=1), 1, atol=1e-14)
            np.testing.assert_allclose(result[f"hist_xy_{kind}"].sum(axis=(1, 2)), 1, atol=1e-14)

    def test_scale_and_global_phase_invariance(self):
        psi, u = target_vectors()
        kwargs = dict(seeds=seeds(), samples_per_repeat=101, batch_size=32)
        original = observe(psi, u, **kwargs)
        changed = observe(np.exp(-.23j) * psi, 2 * np.exp(.79j) * u, **kwargs)
        for key in ("R_p", "R_h", "purity_p", "purity_h", "tv", "hist_n_p", "hist_xy_p", "ess_p"):
            np.testing.assert_allclose(original[key], changed[key], rtol=2e-12, atol=1e-13)
        np.testing.assert_allclose(changed["mass_p"], 8 * original["mass_p"])
        np.testing.assert_allclose(changed["cost_p"], 64 * original["cost_p"])

    def test_streaming_matches_direct_small_sample_reduction(self):
        psi, u = target_vectors()
        repeat_seeds = seeds()
        result = observe(psi, u, seeds=repeat_seeds, samples_per_repeat=23, batch_size=7)
        for block, seed in enumerate(repeat_seeds):
            g = sample_in_batches(seed, 23, 7)
            weights = {"p": np.abs(g.conj() @ u.T)**3,
                       "h": np.abs(g.conj() @ psi.T)**3}
            probabilities = np.abs(g)**2
            number = probabilities @ OCCUPATIONS
            for kind, w in weights.items():
                normalized = w / w.sum(axis=0)
                direct_R = np.einsum("bp,bi,bj->pij", normalized, g, g.conj())
                np.testing.assert_allclose(result[f"blocks_R_{kind}"][block], direct_R, atol=2e-15)
                direct_purity = np.einsum("pij,pji->p", direct_R, direct_R).real
                np.testing.assert_allclose(result[f"blocks_purity_{kind}"][block], direct_purity)
                np.testing.assert_allclose(result[f"blocks_mass_{kind}"][block], 35*w.mean(axis=0))
                np.testing.assert_allclose(result[f"blocks_mean_n_{kind}"][block], number @ normalized)
                np.testing.assert_allclose(result[f"blocks_ess_{kind}"][block],
                                           1 / np.sum(normalized**2, axis=0))
            direct_tv = .5*np.sum(np.abs(weights["p"]/weights["p"].sum(axis=0)
                                          -weights["h"]/weights["h"].sum(axis=0)), axis=0)
            np.testing.assert_allclose(result["blocks_tv"][block], direct_tv, atol=2e-15)
        for kind in ("p", "h"):
            matrices = result[f"blocks_R_{kind}"]
            cross = np.mean([np.einsum("pij,pji->p", matrices[a], matrices[b]).real
                             for a in range(3) for b in range(a+1, 3)], axis=0)
            np.testing.assert_allclose(result[f"purity_cross_{kind}"], cross, atol=1e-15)
            np.testing.assert_allclose(result[f"R_{kind}"], matrices.mean(axis=0))
            direct_se = np.std(result[f"blocks_purity_{kind}"], axis=0, ddof=1)/np.sqrt(3)
            np.testing.assert_allclose(result[f"purity_{kind}_se"], direct_se)
            spectra = np.linalg.eigvalsh(matrices)[..., ::-1]
            np.testing.assert_allclose(result[f"spectrum_{kind}"], spectra.mean(axis=0))
            np.testing.assert_allclose(result[f"spectrum_{kind}_se"], spectra.std(axis=0, ddof=1)/np.sqrt(3))

    def test_pullback_histograms_match_direct_coordinates(self):
        psi, u = target_vectors()
        pullbacks = np.repeat(np.eye(8, dtype=complex)[None, :, :], 2, axis=0)
        # A pair rotation mixes vacuum/pair and complementary pair/full. Its
        # nontrivial number distribution catches transpose/conjugation errors.
        angle, phase = .39, .61
        rotation = np.array([[np.cos(angle), -np.exp(-1j*phase)*np.sin(angle)],
                             [np.exp(1j*phase)*np.sin(angle), np.cos(angle)]])
        for indices in ([0, 1], [6, 7]):
            pullbacks[1][np.ix_(indices, indices)] = rotation
        repeat_seeds = seeds()
        result = observe(psi, u, seeds=repeat_seeds, samples_per_repeat=29,
                         batch_size=11, pullbacks=pullbacks)
        for coordinate in ("n", "xy"):
            for kind in ("p", "h"):
                np.testing.assert_array_equal(result[f"hist_{coordinate}_{kind}"][0],
                                              result[f"hist_{coordinate}_pulled_{kind}"][0])
        direct = {f"{coordinate}_{kind}": [] for coordinate in ("n", "xy") for kind in ("p", "h")}
        for seed in repeat_seeds:
            g = sample_in_batches(seed, 29, 11)
            pulled = g @ pullbacks[1].T
            probabilities = np.abs(pulled)**2
            for kind, vector in (("p", u[1]), ("h", psi[1])):
                weight = np.abs(g.conj() @ vector)**3
                weight /= weight.sum()
                direct[f"n_{kind}"].append(np.histogram(probabilities @ OCCUPATIONS,
                                                        bins=N_EDGES, weights=weight)[0])
                direct[f"xy_{kind}"].append(np.histogram2d(probabilities[:, 0], probabilities[:, 7],
                                                           bins=(XY_EDGES, XY_EDGES), weights=weight)[0])
        for key, values in direct.items():
            coordinate, kind = key.split("_")
            np.testing.assert_allclose(result[f"hist_{coordinate}_pulled_{kind}"][1],
                                       np.mean(values, axis=0), atol=2e-15)

    def test_budget_and_invalid_inputs(self):
        psi, u = target_vectors()
        class Budget:
            calls = 0
            def check(self):
                self.calls += 1
        budget = Budget()
        observe(psi, u, seeds=seeds(), samples_per_repeat=9, batch_size=4, budget=budget)
        self.assertGreaterEqual(budget.calls, 2 * 3 * 3)
        for kwargs in ({"samples_per_repeat": 0}, {"batch_size": 8193}, {"batch_size": True},
                       {"pullbacks": np.zeros((2, 8, 8))}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    observe(psi, u, seeds=seeds(), **kwargs)
        with self.assertRaises(ValueError):
            observe(psi, np.zeros_like(u), seeds=seeds())
        with self.assertRaises(ValueError):
            observe(psi, u, seeds=seeds()[:1])


if __name__ == "__main__":
    unittest.main()
