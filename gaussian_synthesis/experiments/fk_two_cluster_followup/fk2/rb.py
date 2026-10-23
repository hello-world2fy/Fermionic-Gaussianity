"""Paired raw and symmetry-orbit reconstruction for frozen FK witnesses.

For invariant u, the coefficient of g is unchanged on its diagonal Spin(7)
orbit.  Replacing g by Pi g therefore integrates out that orbit analytically:
Pi g = integral U(h)g dh.  Pi g is neither a Gaussian ket nor a normalized
free atom.  Its orbit representation has coefficient cost at most one.  This
module always retains the raw 128-dimensional reconstruction alongside the
orbit reconstruction, and never changes the first experiment's acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

from fk.dictionaries import Dictionary
from fk.observables import covariance
from fk.runtime import ResourceLimit
from fk.validation import gaussian_probe_reference


def _digest(vector):
    return hashlib.sha256(np.asarray(vector, dtype=np.complex128).tobytes()).hexdigest()


def _project(basis, vector):
    return np.einsum("ij,...i->...j", basis.conj(), vector, optimize=False)


def _embed(basis, coordinates):
    return np.einsum("ij,...j->...i", basis, coordinates, optimize=False)


@dataclass(frozen=True)
class Candidate:
    label: str
    t: float | None
    k: int
    u: np.ndarray
    target: np.ndarray
    sourcekind: str
    source_witness_sha256: str
    expected_reconstruction_known: bool = False
    expected_q: float | None = None
    expected_mass: float | None = None
    exact_reconstruction: np.ndarray | None = None
    exact_reference_kind: str | None = None


def _validate_candidate(candidate, basis):
    for name, value in (("witness", candidate.u), ("target", candidate.target)):
        if value.shape != (128,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{candidate.label}: {name} must be a finite 128-vector")
        outside = np.linalg.norm(value - _embed(basis, _project(basis, value)))
        if outside > 1e-10*max(1., np.linalg.norm(value)):
            raise ValueError(f"{candidate.label}: {name} is outside the invariant subspace")
    if not isinstance(candidate.k, (int, np.integer)) or isinstance(candidate.k, bool) or candidate.k < 1:
        raise ValueError("candidate order must be a positive integer")
    if not np.any(candidate.u):
        raise ValueError("candidate witness must be nonzero")
    candidate.u.flags.writeable = False
    candidate.target.flags.writeable = False
    if candidate.exact_reconstruction is not None:
        if (candidate.exact_reconstruction.shape != (128,)
                or not np.all(np.isfinite(candidate.exact_reconstruction))):
            raise ValueError("Exact reconstruction reference must be a finite 128-vector")
        candidate.exact_reconstruction.flags.writeable = False


def _load_frozen(model, source_dir, name, seed, t_values, orders, train_samples):
    source = Path(source_dir)
    json_path = source / f"pilot_{name}_{seed}.json"
    npz_path = source / f"witnesses_{name}_{seed}.npz"
    document = json.loads(json_path.read_text(encoding="utf-8"))
    candidates = []
    with np.load(npz_path, allow_pickle=False) as archive:
        for t in t_values:
            for k in orders:
                matches = [row for row in document["rows"]
                           if row["dictionary"] == name and row["seed"] == seed
                           and float(row["t"]) == float(t) and row["k"] == k
                           and row["train_samples"] == train_samples]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one frozen witness for {name}, seed={seed}, t={t}, k={k}")
                row = matches[0]
                if not row.get("optimizer_converged", False):
                    raise ValueError("Frozen source optimizer did not converge")
                u = np.array(archive[row["key"] + "_u"], dtype=np.complex128, copy=True)
                digest = _digest(u)
                if digest != row["witness_sha256"]:
                    raise ValueError(f"Source witness hash mismatch: {row['key']}")
                candidate = Candidate(row["key"], float(t), int(k), u,
                                      np.array(model.target(float(t)), dtype=np.complex128, copy=True),
                                      "frozen_pilot", digest)
                _validate_candidate(candidate, model.S)
                candidates.append(candidate)
    return candidates, {
        "directory": str(source.resolve()),
        "pilot_file": json_path.name,
        "pilot_file_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "witness_file": npz_path.name,
        "witness_file_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "source_train_samples": int(train_samples),
    }


def hopping_gaussian_probe(model):
    """Derive the unique hopping ground state and verify its invariants."""
    hopping = np.asarray(model.h_hop)[np.ix_(model.even_masks, model.even_masks)]
    energies, eigenvectors = eigh(hopping, subset_by_index=(0, 1), check_finite=True)
    probe = np.asarray(eigenvectors[:, 0], dtype=np.complex128).copy()
    pivot = int(np.argmax(np.abs(probe)))
    probe *= np.conjugate(probe[pivot])/abs(probe[pivot])
    probe /= np.linalg.norm(probe)
    cov = covariance(probe, model)
    purity = np.einsum("ab,bc->ac", cov, cov, optimize=False) + np.eye(16)
    invariance_error = float(np.linalg.norm(probe - _embed(model.S, _project(model.S, probe))))
    purity_error = float(np.linalg.norm(purity))
    metadata = {"hopping_ground_energy": float(energies[0]),
                "hopping_gap_in_even_sector": float(energies[1] - energies[0]),
                "invariance_error": invariance_error, "gaussian_purity_error": purity_error,
                "probe_sha256": _digest(probe)}
    if (invariance_error > 1e-10 or purity_error > 1e-10
            or energies[1]-energies[0] <= 1e-10):
        raise ValueError("Hopping Gaussian probe failed uniqueness, invariance, or purity check")
    return probe, metadata


class PairedAccumulator:
    """Streaming paired means and covariance traces; no 128x128 covariance."""

    def __init__(self, dictionary, basis, candidates):
        self.dictionary = dictionary
        self.basis = np.asarray(basis, dtype=np.complex128)
        if self.basis.shape != (128, 3):
            raise ValueError("FK invariant basis must have shape (128, 3)")
        gram = np.einsum("ij,ik->jk", self.basis.conj(), self.basis, optimize=False)
        if not np.allclose(gram, np.eye(3), atol=1e-11, rtol=0):
            raise ValueError("Invariant basis must be orthonormal")
        self.candidates = list(candidates)
        self.u = np.asarray([candidate.u for candidate in candidates])
        self.orders = np.asarray([candidate.k for candidate in candidates], dtype=int)
        self.d = np.exp([dictionary.log_frame_constant(int(k)) for k in self.orders])
        size = len(candidates)
        self.count = 0
        self.raw_sum = np.zeros((size, 128), dtype=np.complex128)
        self.projected_sum = np.zeros((size, 3), dtype=np.complex128)
        self.raw_norm2_sum = np.zeros(size)
        self.projected_norm2_sum = np.zeros(size)
        self.mass_sum = np.zeros(size)
        self.mass_norm2_sum = np.zeros(size)
        self.max_mass = np.zeros(size)
        self.q_sum = np.zeros(size)

    def add(self, kets):
        kets = np.asarray(kets, dtype=np.complex128)
        if kets.ndim != 2 or kets.shape[1] != 128 or len(kets) == 0:
            raise ValueError("Expected nonempty (N,128) Gaussian batch")
        norm2 = np.sum(np.abs(kets)**2, axis=1)
        if not np.all(np.isfinite(kets)) or not np.allclose(norm2, 1., atol=1e-11, rtol=0):
            raise ValueError("Gaussian atoms must remain normalized")
        overlap = np.einsum("ni,mi->nm", kets.conj(), self.u, optimize=False)
        absolute = np.abs(overlap)
        coefficients = self.d[None, :]*absolute**(2*self.orders[None, :]-2)*overlap
        mass = np.abs(coefficients)
        projected = np.einsum("ni,ij->nj", kets, self.basis.conj(), optimize=False)
        projected_norm2 = np.sum(np.abs(projected)**2, axis=1)
        self.raw_sum += np.einsum("nm,ni->mi", coefficients, kets, optimize=False)
        self.projected_sum += np.einsum("nm,nj->mj", coefficients, projected, optimize=False)
        self.raw_norm2_sum += np.einsum("nm,n->m", mass**2, norm2, optimize=False)
        self.projected_norm2_sum += np.einsum("nm,n->m", mass**2, projected_norm2, optimize=False)
        self.mass_sum += np.sum(mass, axis=0)
        self.mass_norm2_sum += np.sum(mass**2, axis=0)
        self.max_mass = np.maximum(self.max_mass, np.max(mass, axis=0))
        self.q_sum += np.sum(mass*absolute, axis=0)
        self.count += len(kets)

    def means(self):
        if self.count == 0:
            raise ValueError("No samples accumulated")
        raw = self.raw_sum/self.count
        coordinates = self.projected_sum/self.count
        return raw, _embed(self.basis, coordinates), coordinates

    def snapshot(self):
        return {"completed_samples": np.array(self.count), "witnesses": self.u.copy(),
                "orders": self.orders.copy(), "basis": self.basis.copy(),
                "raw_sum": self.raw_sum.copy(), "projected_sum": self.projected_sum.copy(),
                "raw_norm2_sum": self.raw_norm2_sum.copy(),
                "projected_norm2_sum": self.projected_norm2_sum.copy(),
                "mass_sum": self.mass_sum.copy(), "mass_norm2_sum": self.mass_norm2_sum.copy(),
                "max_mass_weight": self.max_mass.copy(), "q_sum": self.q_sum.copy()}

    def row(self, index, tolerance):
        if self.count < 2:
            raise ValueError("At least two samples are needed for variance estimates")
        candidate = self.candidates[index]
        raw, orbit, coordinates = self.means()
        raw, orbit, coordinates = raw[index], orbit[index], coordinates[index]
        raw_residual = float(np.linalg.norm(raw-candidate.target))
        projected_residual = float(np.linalg.norm(coordinates-_project(self.basis, candidate.target)))
        orbit_residual = float(np.linalg.norm(orbit-candidate.target))
        complement = float(np.linalg.norm(raw-orbit))
        raw_variance = max(0., (self.raw_norm2_sum[index]
                               - self.count*float(np.vdot(raw, raw).real))/(self.count-1))
        orbit_variance = max(0., (self.projected_norm2_sum[index]
                                 - self.count*float(np.vdot(coordinates, coordinates).real))/(self.count-1))
        mass = self.mass_sum[index]/self.count
        mass_variance = max(0., (self.mass_norm2_sum[index]-self.count*mass*mass)/(self.count-1))
        row = {
            "candidate": candidate.label, "sourcekind": candidate.sourcekind,
            "t": candidate.t, "k": candidate.k, "samples": self.count,
            "source_witness_sha256": candidate.source_witness_sha256,
            "target_sha256": _digest(candidate.target),
            "expected_reconstruction_known": candidate.expected_reconstruction_known,
            "raw_full_residual": raw_residual,
            "projected_residual": projected_residual,
            "orbit_full_residual": orbit_residual,
            "complement_residual": complement,
            "paired_pythagoras_error": abs(raw_residual**2-orbit_residual**2-complement**2),
            "projection_commutation_error": float(np.linalg.norm(_project(self.basis, raw)-coordinates)),
            "q_estimate": float(self.q_sum[index]/self.count), "mass_estimate": float(mass),
            "mass_standard_error_estimate": float(math.sqrt(mass_variance/self.count)),
            "mass_ess": float(self.mass_sum[index]**2/self.mass_norm2_sum[index]) if self.mass_norm2_sum[index] > 0 else 0.,
            "max_mass_weight_share": float(self.max_mass[index]/self.mass_sum[index]) if self.mass_sum[index] > 0 else 0.,
            "raw_sample_variance_trace": float(raw_variance),
            "orbit_sample_variance_trace": float(orbit_variance),
            "complement_sample_variance_trace": float(max(0., raw_variance-orbit_variance)),
            "paired_variance_ratio": float(raw_variance/orbit_variance) if orbit_variance > 0 else None,
            "variance_reduction_fraction": float(1-orbit_variance/raw_variance) if raw_variance > 0 else None,
            "raw_mean_rmse_estimate": float(math.sqrt(raw_variance/self.count)),
            "orbit_mean_rmse_estimate": float(math.sqrt(orbit_variance/self.count)),
            "raw_within_tolerance": raw_residual <= tolerance,
            "orbit_within_tolerance": orbit_residual <= tolerance,
            "reconstruction_tolerance": float(tolerance),
            "raw_fock_repaired_cost_numeric": float(mass+np.sum(np.abs(candidate.target-raw))),
            "orbit_fock_repaired_cost_numeric": float(mass+np.sum(np.abs(candidate.target-orbit))),
            "statistical_status": "paired_monte_carlo_diagnostic_no_formal_certificate",
            "formally_certified": False, "original_first_round_acceptance_changed": False,
            "exact_reference_kind": candidate.exact_reference_kind,
            "exact_bias_l2": None, "raw_error_to_exact_C": None,
            "orbit_error_to_exact_C": None, "rb_error_to_exact_C": None,
        }
        if candidate.exact_reconstruction is not None:
            bias = candidate.exact_reconstruction-candidate.target
            raw_noise = raw-candidate.exact_reconstruction
            orbit_noise = orbit-candidate.exact_reconstruction
            row.update(exact_bias_l2=float(np.linalg.norm(bias)),
                       raw_error_to_exact_C=float(np.linalg.norm(raw_noise)),
                       orbit_error_to_exact_C=float(np.linalg.norm(orbit_noise)),
                       rb_error_to_exact_C=float(np.linalg.norm(orbit_noise)),
                       raw_bias_noise_cross_term=float(2*np.vdot(raw_noise, bias).real),
                       orbit_bias_noise_cross_term=float(2*np.vdot(orbit_noise, bias).real))
        if candidate.expected_q is not None:
            row.update(expected_q=float(candidate.expected_q),
                       q_absolute_error=abs(row["q_estimate"]-candidate.expected_q))
        if candidate.expected_mass is not None:
            row.update(expected_mass=float(candidate.expected_mass),
                       mass_absolute_error=abs(row["mass_estimate"]-candidate.expected_mass))
        if candidate.sourcekind == "hopping_gaussian_probe":
            reference = gaussian_probe_reference(self.dictionary, candidate.k)
            row["raw_rmse_exact"] = math.sqrt((reference["coefficient_second_moment"]-1)/self.count)
            row["q_standard_error_exact"] = math.sqrt((reference["q_second_moment"]-1)/self.count)
            row["mass_standard_error_exact"] = math.sqrt(
                (reference["coefficient_second_moment"]-reference["mass"]**2)/self.count)
            row["probe_diagnostic_calibration_passed"] = bool(
                abs(row["q_estimate"]-reference["q"]) <= 5*row["q_standard_error_exact"]+1e-12
                and abs(row["mass_estimate"]-reference["mass"]) <= 5*row["mass_standard_error_exact"]+1e-12
                and row["raw_full_residual"] <= 6*row["raw_rmse_exact"]+1e-12)
            row["probe_diagnostic_rule"] = "predeclared 5 exact scalar SE for q/B and 6 exact raw reconstruction RMSE; not a formal confidence guarantee"
        return row


def run_rb(model, source_dir, config, budget, checkpoint=None, extra_witnesses=None):
    """Run one (dictionary, seed) stage using independent paired validation.

    Required config keys: dictionary, seed.  Optional: t_values=[0,11,20],
    orders=[2,4], sample_sizes=[4096,16384,65536], batch_size=256,
    source_train_samples=65536, rng_purpose=800, reconstruction_tolerance=.01.
    Checkpoint receives (report, arrays) at every requested sample-size boundary.
    Optional deterministic candidates have label,t,k,u,expected_reconstruction;
    optional expected_q/expected_mass are diagnostic references, not assumed.
    No source files are written and no source witness is reoptimized.
    """
    name, seed = str(config["dictionary"]), int(config["seed"])
    dictionary = Dictionary(name)
    purpose = int(config.get("rng_purpose", 800))
    if purpose < 800:
        raise ValueError("Follow-up validation requires an independent RNG purpose >= 800")
    sizes = [int(n) for n in config.get("sample_sizes", [4096, 16384, 65536])]
    if not sizes or sizes != sorted(set(sizes)) or sizes[0] < 2:
        raise ValueError("sample_sizes must be increasing unique integers >= 2")
    orders = [int(k) for k in config.get("orders", [2, 4])]
    t_values = list(config.get("t_values", [0, 11, 20]))
    batch_size = int(config.get("batch_size", 256))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tolerance = float(config.get("reconstruction_tolerance", .01))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("reconstruction_tolerance must be positive and finite")
    budget.check()
    candidates, provenance = _load_frozen(model, source_dir, name, seed, t_values, orders,
                                          int(config.get("source_train_samples", 65536)))
    probe_metadata = None
    if name == "full":
        probe, probe_metadata = hopping_gaussian_probe(model)
        for k in orders:
            reference = gaussian_probe_reference(dictionary, k)
            candidates.append(Candidate(f"hopping_probe_k{k}", None, k, probe.copy(), probe.copy(),
                                        "hopping_gaussian_probe", _digest(probe), True,
                                        reference["q"], reference["mass"], probe.copy(),
                                        "analytic_gaussian_probe_frame_identity"))
    for extra in extra_witnesses or []:
        if name != "full" or int(extra["k"]) != 2:
            raise ValueError("Deterministic extras currently require the full dictionary and k=2")
        u = np.array(extra["u"], dtype=np.complex128, copy=True)
        target = np.array(extra["expected_reconstruction"], dtype=np.complex128, copy=True)
        candidates.append(Candidate(str(extra["label"]), float(extra["t"]), 2, u, target,
                                    "deterministic_k2", _digest(u), True,
                                    extra.get("expected_q"), extra.get("expected_mass")))
    exact_reference = {"status": "not_applicable_to_product"}
    if name == "full" and not config.get("enable_exact_reference",True):
        exact_reference = {"status": "disabled_unverified_reference_stage"}
    if (name == "full" and config.get("enable_exact_reference",True)
            and any(candidate.k == 2 for candidate in candidates)):
        try:
            from .exact_moments import ExactK2Moments
        except ImportError as error:
            exact_reference = {"status": "unavailable", "reason": str(error)}
        else:
            budget.check()
            backend = ExactK2Moments(model)
            exact_reference = {"status": "available", "diagnostics": backend.diagnostics}
            for index, candidate in enumerate(candidates):
                if candidate.k == 2:
                    z = _project(model.S, candidate.u)
                    exact_c = np.array(backend.reconstruction(z), dtype=np.complex128, copy=True)
                    q = backend.evaluate(z, hessian=False).q
                    candidates[index] = replace(candidate, expected_reconstruction_known=True,
                                                expected_q=float(q), exact_reconstruction=exact_c,
                                                exact_reference_kind="deterministic_full_k2_cartan_projector")
    if len({candidate.label for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate labels must be unique within a stage")
    for candidate in candidates:
        _validate_candidate(candidate, model.S)
    labels = [seed, int(name == "product"), purpose]
    rng = np.random.default_rng(np.random.SeedSequence(labels))
    accumulator = PairedAccumulator(dictionary, model.S, candidates)
    report = {
        "status": "running", "dictionary": name, "seed": seed,
        "rng_seed_sequence": labels, "source": provenance,
        "planned_sample_sizes": sizes, "candidate_count": len(candidates), "rows": [],
        "probe": probe_metadata,
        "exact_k2_reference": exact_reference,
        "product_probe_status": "not_used_no_valid_analytic_hopping_probe_for_product_dictionary" if name == "product" else "not_applicable",
        "representation": "Pi g is an unnormalized diagonal Spin(7) Gaussian orbit average with cost <= 1; it is never treated as a normalized Gaussian atom",
        "paired_estimator_mean": "For invariant frozen witnesses, raw C and orbit-averaged C have the same exact Haar expectation",
        "formal_confidence_guarantee": False,
        "variance_estimates_are_confidence_intervals": False,
        "original_source_files_modified": False,
    }
    arrays = {"targets": np.asarray([candidate.target for candidate in candidates]),
              "candidate_labels": np.asarray([candidate.label for candidate in candidates]),
              "exact_reconstruction_available": np.asarray([candidate.exact_reconstruction is not None for candidate in candidates]),
              "exact_reconstructions": np.asarray([candidate.exact_reconstruction
                    if candidate.exact_reconstruction is not None else np.zeros(128, complex)
                    for candidate in candidates])}
    try:
        for size in sizes:
            while accumulator.count < size:
                budget.check()
                batch = dictionary.sample(min(batch_size, size-accumulator.count), rng)
                accumulator.add(batch.kets)
            budget.check()
            for index in range(len(candidates)):
                row = accumulator.row(index, tolerance)
                row.update(dictionary=name, seed=seed)
                report["rows"].append(row)
            raw, orbit, projected = accumulator.means()
            arrays[f"n{size}_raw_C"] = raw.copy()
            arrays[f"n{size}_orbit_C"] = orbit.copy()
            arrays[f"n{size}_projected_C"] = projected.copy()
            arrays.update(accumulator.snapshot())
            report.update(completed_samples=accumulator.count, rng_state=rng.bit_generator.state,
                          usage=budget.usage())
            if checkpoint:
                checkpoint(report, arrays)
        report["status"] = "complete"
    except ResourceLimit as error:
        report.update(status="resource_limit", error=str(error))
    arrays.update(accumulator.snapshot())
    report.update(completed_samples=accumulator.count, rng_state=rng.bit_generator.state,
                  usage=budget.usage())
    probe_rows = [row for row in report["rows"] if row["sourcekind"] == "hopping_gaussian_probe"]
    final_probes = [row for row in probe_rows if row["samples"] == sizes[-1]]
    if name != "full":
        report["probe_calibration_status"] = "not_applicable"
    elif len(final_probes) != len(orders):
        report["probe_calibration_status"] = "incomplete"
    elif all(row["probe_diagnostic_calibration_passed"] for row in final_probes):
        report["probe_calibration_status"] = "passed"
    else:
        report["probe_calibration_status"] = "failed"
        if report["status"] == "complete":
            report["status"] = "probe_calibration_failed"
    report["all_recorded_probe_checks_passed"] = (all(row["probe_diagnostic_calibration_passed"] for row in probe_rows)
                                                   if probe_rows else None)
    return report, arrays

