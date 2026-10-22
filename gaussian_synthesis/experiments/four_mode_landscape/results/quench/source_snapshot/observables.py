"""Streaming diagnostics for the k=2 Gaussian configuration distributions.

All histograms contain probability mass per bin, not probability density.
The x/y coordinates are the vacuum/full-occupation probabilities of a Gaussian
configuration. Standard errors describe independent Monte Carlo repetitions;
they are not rigorous confidence bounds. ESS is reported per repetition.
"""

import _bootstrap  # noqa: F401 -- establish the experiment's import/thread setup
import numpy as np

from gaussian_synthesis.sampling import HaarFourModeSampler


N_EDGES = np.linspace(0.0, 4.0, 41)
XY_EDGES = np.linspace(0.0, 1.0, 41)
OCCUPATIONS = np.array([0, 2, 2, 2, 2, 2, 2, 4], dtype=float)
_MASS_FACTOR = 35.0


def _check(budget):
    if budget is not None:
        budget.check()


def _sampler(seed):
    # Reconstructing PCG64 from the same SeedSequence replays a repetition.
    return HaarFourModeSampler(np.random.Generator(np.random.PCG64(seed)))


def _weights(samples, vectors):
    with np.errstate(over="raise", invalid="raise"):
        return np.abs(samples.conj() @ vectors.T) ** 3


def _coordinates(samples):
    probabilities = np.abs(samples) ** 2
    number = probabilities @ OCCUPATIONS
    # Clipping also places the closed right edge in the final bin.
    nbin = np.clip(np.floor(number * 10).astype(np.int64), 0, 39)
    xbin = np.clip(np.floor(probabilities[:, 0] * 40).astype(np.int64), 0, 39)
    ybin = np.clip(np.floor(probabilities[:, 7] * 40).astype(np.int64), 0, 39)
    return number, nbin, xbin * 40 + ybin


def _histograms(weights, nbin, xybin, hist_n, hist_xy):
    for point in range(weights.shape[1]):
        hist_n[point] += np.bincount(nbin, weights=weights[:, point], minlength=40)
        hist_xy[point] += np.bincount(xybin, weights=weights[:, point], minlength=1600)


def _mean_se(values):
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=1) / np.sqrt(len(values))


def _cross_purity(matrices):
    """Independent-repetition cross purity and its delete-one jackknife SE.

    This removes the within-repetition self-pair term of Tr(Rhat**2). It does
    not remove all bias due to the self-normalized importance weights.
    """
    repetitions = len(matrices)
    pairs = []
    for first in range(repetitions):
        for second in range(first + 1, repetitions):
            pairs.append(np.einsum("pij,pji->p", matrices[first], matrices[second]).real)
    mean = np.mean(pairs, axis=0)
    if repetitions < 3:
        # Two matrices supply one independent cross-product, so no empirical
        # uncertainty estimate is available; omit its SE in the public result.
        return mean, None
    leave_one_out = []
    for omitted in range(repetitions):
        selected = [matrices[j] for j in range(repetitions) if j != omitted]
        terms = [np.einsum("pij,pji->p", selected[i], selected[j]).real
                 for i in range(len(selected)) for j in range(i + 1, len(selected))]
        leave_one_out.append(np.mean(terms, axis=0))
    leave_one_out = np.asarray(leave_one_out)
    center = np.mean(leave_one_out, axis=0)
    se = np.sqrt((repetitions - 1) / repetitions * np.sum((leave_one_out - center) ** 2, axis=0))
    return mean, se


def observe(psi, witnesses, *, seeds, samples_per_repeat=65536,
            batch_size=8192, budget=None, pullbacks=None):
    """Estimate p/h diagnostics on common Haar samples in independent blocks.

    ``psi`` and ``witnesses`` have shape (points, 8). ``seeds`` contains at
    least two independent SeedSequences, normally four. A pullback matrix
    maps a column ket from the laboratory frame to canonical coordinates.
    Only its coordinate histograms are returned, in addition to laboratory
    frame diagnostics. No samples are retained after their batch is reduced.

    ``blocks_*`` keeps each repetition's raw scalar metrics and R matrices.
    Non-block keys contain means, and ``*_se`` denotes a repetition-based
    standard error. R is the equal-weight mean of independently normalized
    block matrices, not a ratio of their pooled raw sums. ``spectrum_*`` is
    the mean sorted block spectrum; ``spectrum_of_mean_R_*`` is the spectrum
    of the mean matrix itself. TV uses a second pass replaying each seed after
    its two normalization integrals are known.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    witnesses = np.asarray(witnesses, dtype=np.complex128)
    if psi.ndim != 2 or psi.shape[1:] != (8,) or len(psi) == 0 or witnesses.shape != psi.shape:
        raise ValueError("psi and witnesses must have matching nonempty shape (points, 8)")
    if not np.all(np.isfinite(psi)) or not np.all(np.isfinite(witnesses)):
        raise ValueError("State vectors must be finite")
    if np.any(np.linalg.norm(psi, axis=1) == 0) or np.any(np.linalg.norm(witnesses, axis=1) == 0):
        raise ValueError("Distribution vectors must be nonzero")
    seeds = tuple(seeds)
    if len(seeds) < 2:
        raise ValueError("At least two independent repetition seeds are required")
    for name, value in (("samples_per_repeat", samples_per_repeat), ("batch_size", batch_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if batch_size > HaarFourModeSampler.max_batch:
        raise ValueError("batch_size cannot exceed 8192")
    points = len(psi)
    if pullbacks is not None:
        pullbacks = np.asarray(pullbacks, dtype=np.complex128)
        if pullbacks.shape != (points, 8, 8) or not np.all(np.isfinite(pullbacks)):
            raise ValueError("pullbacks must have finite shape (points, 8, 8)")
        gram = np.swapaxes(pullbacks.conj(), -1, -2) @ pullbacks
        if not np.allclose(gram, np.eye(8), atol=1e-10, rtol=0):
            raise ValueError("pullbacks must be unitary")

    blocks = {}
    hist_blocks = {f"hist_{coordinate}_{kind}": []
                   for coordinate in ("n", "xy") for kind in ("p", "h")}
    if pullbacks is not None:
        hist_blocks.update({f"hist_{coordinate}_pulled_{kind}": []
                            for coordinate in ("n", "xy") for kind in ("p", "h")})

    for seed in seeds:
        _check(budget)
        totals = {kind: {
            "weight": np.zeros(points), "square": np.zeros(points),
            "number": np.zeros(points), "R": np.zeros((points, 8, 8), dtype=complex),
            "hist_n": np.zeros((points, 40)), "hist_xy": np.zeros((points, 1600)),
        } for kind in ("p", "h")}
        if pullbacks is not None:
            for total in totals.values():
                total["hist_n_pulled"] = np.zeros((points, 40))
                total["hist_xy_pulled"] = np.zeros((points, 1600))
        sampler = _sampler(seed)
        for offset in range(0, samples_per_repeat, batch_size):
            _check(budget)
            samples = sampler.sample(min(batch_size, samples_per_repeat - offset))
            number, nbin, xybin = _coordinates(samples)
            projectors = (samples[:, :, None] * samples[:, None, :].conj()).reshape(len(samples), 64)
            weights = {"p": _weights(samples, witnesses), "h": _weights(samples, psi)}
            for kind, weight in weights.items():
                total = totals[kind]
                total["weight"] += np.sum(weight, axis=0)
                total["square"] += np.einsum("bp,bp->p", weight, weight)
                total["number"] += number @ weight
                total["R"] += (weight.T @ projectors).reshape(points, 8, 8)
                _histograms(weight, nbin, xybin, total["hist_n"], total["hist_xy"])
            if pullbacks is not None:
                for point in range(points):
                    _, pulled_n, pulled_xy = _coordinates(samples @ pullbacks[point].T)
                    for kind, weight in weights.items():
                        total = totals[kind]
                        total["hist_n_pulled"][point] += np.bincount(
                            pulled_n, weights=weight[:, point], minlength=40)
                        total["hist_xy_pulled"][point] += np.bincount(
                            pulled_xy, weights=weight[:, point], minlength=1600)

        normalizations = {}
        for kind, total in totals.items():
            denominator = total["weight"]
            if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
                raise FloatingPointError("Distribution normalization is zero or nonfinite")
            normalizations[kind] = denominator / samples_per_repeat
            matrix = total["R"] / denominator[:, None, None]
            mass = _MASS_FACTOR * normalizations[kind]
            values = {
                f"R_{kind}": matrix,
                f"purity_{kind}": np.einsum("pij,pji->p", matrix, matrix).real,
                f"mass_{kind}": mass,
                f"cost_{kind}": mass ** 2,
                f"mean_n_{kind}": total["number"] / denominator,
                f"ess_{kind}": denominator ** 2 / total["square"],
            }
            for key, value in values.items():
                blocks.setdefault(key, []).append(value)
            for coordinate in ("n", "xy"):
                histogram = total[f"hist_{coordinate}"] / denominator[:, None]
                shape = (points, 40) if coordinate == "n" else (points, 40, 40)
                hist_blocks[f"hist_{coordinate}_{kind}"].append(histogram.reshape(shape))
                if pullbacks is not None:
                    histogram = total[f"hist_{coordinate}_pulled"] / denominator[:, None]
                    hist_blocks[f"hist_{coordinate}_pulled_{kind}"].append(histogram.reshape(shape))

        # Exactly replay this repetition. The denominator estimates came from
        # its first pass; no O(samples * points) storage is needed.
        sampler = _sampler(seed)
        tv = np.zeros(points)
        for offset in range(0, samples_per_repeat, batch_size):
            _check(budget)
            samples = sampler.sample(min(batch_size, samples_per_repeat - offset))
            p = _weights(samples, witnesses) / normalizations["p"]
            h = _weights(samples, psi) / normalizations["h"]
            tv += np.sum(np.abs(p - h), axis=0)
        blocks.setdefault("tv", []).append(tv / (2 * samples_per_repeat))

    result = {"n_edges": N_EDGES.copy(), "xy_edges": XY_EDGES.copy(),
              "samples_per_repeat": np.asarray(samples_per_repeat),
              "repetitions": np.asarray(len(seeds))}
    for key, values in blocks.items():
        values = np.asarray(values)
        result[f"blocks_{key}"] = values
        if key.startswith("R_"):
            result[key] = np.mean(values, axis=0)
        else:
            result[key], result[f"{key}_se"] = _mean_se(values)
    for key, values in hist_blocks.items():
        values = np.asarray(values)
        mean, se = _mean_se(values)
        result[key] = mean
        if key.startswith("hist_n_"):
            result[f"{key}_se"] = se
    for kind in ("p", "h"):
        spectra = np.linalg.eigvalsh(result[f"blocks_R_{kind}"])[..., ::-1]
        result[f"blocks_spectrum_{kind}"] = spectra
        result[f"spectrum_{kind}"], result[f"spectrum_{kind}_se"] = _mean_se(spectra)
        result[f"spectrum_of_mean_R_{kind}"] = np.linalg.eigvalsh(result[f"R_{kind}"])[:, ::-1]
        cross, se = _cross_purity(result[f"blocks_R_{kind}"])
        result[f"purity_cross_{kind}"] = cross
        if se is not None:
            result[f"purity_cross_{kind}_se"] = se
    _check(budget)
    return result
