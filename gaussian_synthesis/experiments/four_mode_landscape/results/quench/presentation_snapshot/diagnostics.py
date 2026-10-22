"""Replay saved Haar streams to diagnose histogram regression flags.

This is a paired covariance check, not a new independent statistical test.
It never replaces the original experiment acceptance result or its samples.
"""

import _bootstrap
import argparse
import json
from pathlib import Path
import time
import numpy as np
from gaussian_synthesis.sampling import HaarFourModeSampler
from gaussian_synthesis.solver import ResourceBudget
from reference import cat_state, cat_witness


_SAMPLE_LEVEL_METHOD = (
    "Approximate self-normalized ratio delta-method SE, using exactly replayed "
    "saved samples. For each independent repetition: Var(Hhat) = "
    "N/(N-1) * sum_i[w_i^2 (I_i-Hhat)^2] / (sum_i w_i)^2. "
    "The variance of the mean of B repetitions is sum_b Var(Hhat_b)/B^2; "
    "variances from the independent direct and canonical streams are added. "
    "This post-run diagnostic is not a confidence certificate and does not "
    "replace the original regression threshold or acceptance result."
)


def _sample_level_histogram_errors(config, vectors, flagged, budget):
    """Replay frozen streams for ratio-estimator variances at flagged bins.

    The denominator is (sum w)**2, not sum(w**2). Centering the per-sample
    influence by the block's own Hhat makes its empirical mean zero. The
    N/(N-1) multiplier is the finite-N sample-variance correction. Each block
    remains separately normalized, matching observe()'s equal-block average.
    """
    groups = sorted({(row["parameter_index"], row["bin_index"]) for row in flagged})
    if not groups:
        return {}, {}
    count = config["samples_per_repeat"]
    if count < 2:
        raise ValueError("The sample-level variance diagnostic requires at least two samples per repetition")
    means, variances = {}, {}
    for stream_name in ("observations", "reference"):
        block_histograms, block_variances = [], []
        for description in config["rng_streams"][stream_name]:
            seed = np.random.SeedSequence(description["entropy"], spawn_key=tuple(description["spawn_key"]))
            sampler = HaarFourModeSampler(np.random.Generator(np.random.PCG64(seed)))
            # Last axis: sum w, sum w I, sum w**2, sum w**2 I.
            sums = np.zeros((2, len(groups), 4))
            for start in range(0, count, config["batch_size"]):
                budget.check()
                g = sampler.sample(min(config["batch_size"], count-start))
                for group, (point, bin_index) in enumerate(groups):
                    if stream_name == "observations":
                        coordinate_kets = g @ vectors["unitaries"][point].conj()
                        targets = (vectors["u"][point], vectors["psi"][point])
                    else:
                        coordinate_kets = g
                        targets = (cat_witness(vectors["r"][point]), cat_state(vectors["r"][point]))
                    number = np.abs(coordinate_kets)**2 @ np.array([0, 2, 2, 2, 2, 2, 2, 4])
                    # Same closed-right-edge convention as observe().
                    bins = np.clip(np.floor(number*10).astype(np.int64), 0, 39)
                    inside = bins == bin_index
                    for label_index, target in enumerate(targets):
                        weight = np.abs(g.conj() @ target)**3
                        square = weight**2
                        sums[label_index, group] += (
                            weight.sum(), weight[inside].sum(), square.sum(), square[inside].sum())
            total, numerator, square, square_inside = np.moveaxis(sums, -1, 0)
            histogram = numerator/total
            # This is exactly sum w_i**2 (I_i-Hhat)**2, split into its
            # nonnegative inside/outside contributions to avoid cancellation.
            centered_square = (square_inside*(1-histogram)**2
                               + np.maximum(0., square-square_inside)*histogram**2)
            variance = count/(count-1)*centered_square/total**2
            block_histograms.append(histogram)
            block_variances.append(variance)
        repetitions = len(block_histograms)
        means[stream_name] = np.mean(block_histograms, axis=0)
        variances[stream_name] = np.sum(block_variances, axis=0)/repetitions**2
    combined = np.sqrt(variances["observations"]+variances["reference"])
    errors, replayed = {}, {}
    for group, (point, bin_index) in enumerate(groups):
        for label_index, label in enumerate(("p", "h")):
            key = (label, point, bin_index)
            errors[key] = float(combined[label_index, group])
            replayed[key] = (float(means["observations"][label_index, group]),
                             float(means["reference"][label_index, group]))
    return errors, replayed


def diagnose_histograms(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config["kind"] != "quench":
        raise ValueError("This diagnostic compares the quench with its canonical static reference")
    with np.load(directory / "vectors.npz", allow_pickle=False) as saved:
        vectors = dict(saved)
    with np.load(directory / "observations.npz", allow_pickle=False) as saved:
        observed = dict(saved)
    with np.load(directory / "reference_observations.npz", allow_pickle=False) as saved:
        predicted = dict(saved)
    flagged = []
    indices = set()
    for label in ("p", "h"):
        difference = np.abs(observed[f"hist_n_pulled_{label}"]-predicted[f"hist_n_{label}"])
        se = np.sqrt(observed[f"hist_n_pulled_{label}_se"]**2+predicted[f"hist_n_{label}_se"]**2)
        tolerance = np.maximum(6*se, .002)
        for i, j in np.argwhere(difference > tolerance):
            indices.add(int(i))
            flagged.append({"distribution": label, "parameter_index": int(i), "phi": float(vectors["parameters"][i]),
                            "bin_index": int(j), "bin_left": float(observed["n_edges"][j]),
                            "bin_right": float(observed["n_edges"][j+1]), "absolute_difference": float(difference[i,j]),
                            "combined_standard_error": float(se[i,j]), "preset_tolerance": float(tolerance[i,j])})
    indices = sorted(indices)
    budget = ResourceBudget(30, 256)
    overlap_error, paired_histogram_error, saved_replay_error = 0., 0., 0.
    per_point = []
    for point in indices:
        budget.check()
        lab_vectors = {"p": vectors["u"][point], "h": vectors["psi"][point]}
        canonical_vectors = {"p": cat_witness(vectors["r"][point]), "h": cat_state(vectors["r"][point])}
        replayed = {label: [] for label in ("p", "h")}
        matched = {label: [] for label in ("p", "h")}
        point_error = 0.
        for description in config["rng_streams"]["observations"]:
            seed = np.random.SeedSequence(description["entropy"], spawn_key=tuple(description["spawn_key"]))
            sampler = HaarFourModeSampler(np.random.Generator(np.random.PCG64(seed)))
            totals = {label: np.zeros((2, 40)) for label in ("p", "h")}
            for start in range(0, config["samples_per_repeat"], config["batch_size"]):
                budget.check()
                g = sampler.sample(min(config["batch_size"], config["samples_per_repeat"]-start))
                canonical_g = g @ vectors["unitaries"][point].conj()
                number = np.abs(canonical_g)**2 @ np.array([0,2,2,2,2,2,2,4])
                for label in ("p", "h"):
                    direct = g.conj() @ lab_vectors[label]
                    canonical = canonical_g.conj() @ canonical_vectors[label]
                    point_error = max(point_error, float(np.max(abs(direct-canonical))))
                    totals[label][0] += np.histogram(number, bins=observed["n_edges"], weights=abs(direct)**3)[0]
                    totals[label][1] += np.histogram(number, bins=observed["n_edges"], weights=abs(canonical)**3)[0]
            for label in ("p", "h"):
                replayed[label].append(totals[label][0]/totals[label][0].sum())
                matched[label].append(totals[label][1]/totals[label][1].sum())
        for label in ("p", "h"):
            direct = np.mean(replayed[label], axis=0)
            canonical = np.mean(matched[label], axis=0)
            paired_histogram_error = max(paired_histogram_error, float(np.max(abs(direct-canonical))))
            saved_replay_error = max(saved_replay_error, float(np.max(abs(direct-observed[f"hist_n_pulled_{label}"][point]))))
        overlap_error = max(overlap_error, point_error)
        per_point.append({"parameter_index": point, "phi": float(vectors["parameters"][point]), "maximum_overlap_error": point_error})
    sample_errors, sample_replays = _sample_level_histogram_errors(config, vectors, flagged, budget)
    sample_replay_error = 0.
    for row in flagged:
        label, point, bin_index = row["distribution"], row["parameter_index"], row["bin_index"]
        key = (label, point, bin_index)
        error = sample_errors[key]
        row["sample_level_combined_se"] = error
        row["difference_over_sample_level_se"] = row["absolute_difference"]/error if error > 0 else None
        row["sample_level_se_method"] = _SAMPLE_LEVEL_METHOD
        direct, canonical = sample_replays[key]
        sample_replay_error = max(sample_replay_error,
            abs(direct-observed[f"hist_n_pulled_{label}"][point, bin_index]),
            abs(canonical-predicted[f"hist_n_{label}"][point, bin_index]))
    result = {"kind": "post_run_paired_covariance_diagnostic", "original_acceptance_unchanged": True,
              "uses_new_random_draws": False, "is_independent_statistical_certificate": False,
              "flagged_bin_count": len(flagged), "tested_bin_count": 2*len(vectors["parameters"])*40,
              "flagged_bins": flagged, "paired_check_points": per_point,
              "maximum_overlap_covariance_error": overlap_error,
              "maximum_paired_histogram_error": paired_histogram_error,
              "maximum_saved_histogram_replay_error": saved_replay_error,
              "maximum_sample_level_histogram_replay_error": sample_replay_error,
              "sample_level_se_method": _SAMPLE_LEVEL_METHOD,
              "sample_level_se_is_approximate": True,
              "sample_level_se_changes_acceptance": False,
              "paired_check_passed": bool(max(overlap_error,paired_histogram_error,saved_replay_error) <= 1e-10),
              "elapsed_seconds": budget.elapsed,
              "interpretation": "Same-sample Gaussian covariance and the saved histogram reduction are checked algebraically. Independent-sample bin differences remain as originally observed; this diagnostic does not erase their regression flags."}
    (directory / "histogram_diagnostics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose_histograms(args.input)
    print(json.dumps({k:v for k,v in result.items() if k not in ("flagged_bins","paired_check_points")},indent=2))
