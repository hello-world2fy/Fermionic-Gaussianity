"""Streaming invariant distributions and paired orbit-average reconstruction.

The scalar coefficient weight is never altered by projection.  Only the ket
factor is Rao--Blackwellized.  Groups depend on orbit invariants, so projecting
each group contribution preserves its exact expectation.  Two interleaved,
independent iid halves give an unbiased unnormalized coherent kernel; Wick
values from the normalized reconstructed ket remain plug-in diagnostics.
"""
from __future__ import annotations

import hashlib
import math
import copy
import numpy as np

from fk.dictionaries import Dictionary
from fk.observables import atom_invariants, exact_metrics
from fk.runtime import ResourceLimit
from fk.validation import formal_budget, gaussian_probe_reference
from fk2.rb import hopping_gaussian_probe


INVARIANTS = ("eta", "mass_a", "mass_b", "mass_dot", "saturation_a", "saturation_b")


def _digest(vector):
    return hashlib.sha256(np.asarray(vector, np.complex128).tobytes()).hexdigest()


def invariant_bins(eta, mass_a):
    """Return fixed row-major mass-A by eta bins, with closed upper edge."""
    x = np.minimum((4*np.clip(eta, 0., 1.)).astype(int), 3)
    y = np.minimum((4*np.clip(mass_a, 0., 1.)).astype(int), 3)
    return 4*y+x


def eta_groups(eta):
    """The predeclared groups are <= .25, (.25,.75], and > .75."""
    return np.searchsorted(np.asarray([.25, .75]), eta, side="left")


def cross_kernel(left, right, operator):
    """Hermitian unbiased cross-half estimate of <phi_a|O|phi_b>."""
    first = left.conj() @ operator @ right.T
    second = right.conj() @ operator @ left.T
    return (first+second)/2


def _prepare_candidates(model, candidates, include_probes):
    prepared = []
    for item in candidates:
        entry = dict(item)
        entry["id"] = str(item["id"])
        entry["k"] = int(item["k"])
        if entry["k"] < 1 or entry["k"] != item["k"]:
            raise ValueError("candidate order must be a positive integer")
        entry["u"] = np.asarray(item["u"], np.complex128).copy()
        entry["target"] = np.asarray(item.get("target", model.target(float(item["t"]))
                if item.get("t") is not None else None), np.complex128).copy()
        entry.setdefault("source", "unspecified")
        entry.setdefault("is_probe", False)
        entry.setdefault("deterministic_verified", False)
        entry.setdefault("optimizer_converged", True)
        prepared.append(entry)
    probe_metadata = None
    if include_probes:
        probe, probe_metadata = hopping_gaussian_probe(model)
        for k in (2, 4):
            prepared.append({"id": f"probe_k{k}", "k": k, "t": None,
                "u": probe.copy(), "target": probe.copy(), "is_probe": True,
                "source": "hopping_gaussian_probe", "deterministic_verified": True,
                "optimizer_converged": True})
    if not prepared or len({p["id"] for p in prepared}) != len(prepared):
        raise ValueError("candidate IDs must be unique and candidates nonempty")
    for entry in prepared:
        for key in ("u", "target"):
            vector = entry[key]
            if vector.shape != (128,) or not np.all(np.isfinite(vector)):
                raise ValueError("candidate vectors must be finite complex 128-vectors")
            projection = model.S @ (model.S.conj().T @ vector)
            if np.linalg.norm(vector-projection) > 1e-10*max(1., np.linalg.norm(vector)):
                raise ValueError("Rao--Blackwellization requires invariant candidates")
        if np.linalg.norm(entry["u"]) == 0:
            raise ValueError("witness cannot be zero")
        if abs(np.linalg.norm(entry["target"])-1.) > 1e-10:
            raise ValueError("target must be normalized")
        entry["witness_sha256"] = _digest(entry["u"])
    return prepared, probe_metadata


class StructureAccumulator:
    """O(candidate count) memory for moments, histograms, and coherent groups."""

    def __init__(self, model, candidates):
        self.model = model
        self.candidates = candidates
        self.dictionary = Dictionary("full")
        self.u = np.asarray([c["u"] for c in candidates])
        self.orders = np.asarray([c["k"] for c in candidates], int)
        self.d = np.exp([self.dictionary.log_frame_constant(int(k)) for k in self.orders])
        m = len(candidates)
        self.count = 0
        self.half_counts = np.zeros(2, int)
        self.raw_sum = np.zeros((m, 128), complex)
        self.rb_sum = np.zeros((m, 3), complex)
        self.rb_norm2_sum = np.zeros(m)
        self.mass_sum = np.zeros(m)
        self.mass_sq_sum = np.zeros(m)
        self.q_sum = np.zeros(m)
        self.q_sq_sum = np.zeros(m)
        self.max_mass = np.zeros(m)
        self.hist_sum = np.zeros((m, 16))
        self.hist_sq_sum = np.zeros((m, 16))
        self.haar_counts = np.zeros(16, int)
        self.inv_sum = np.zeros((m, len(INVARIANTS)))
        self.inv_w2_sum = np.zeros_like(self.inv_sum)
        self.inv_w2x2_sum = np.zeros_like(self.inv_sum)
        self.group_sum = np.zeros((m, 3, 3), complex)
        self.group_half_sum = np.zeros((m, 2, 3, 3), complex)
        self.group_norm2_sum = np.zeros((m, 3))
        self.group_mass_sum = np.zeros((m, 3))
        self.group_mass_sq_sum = np.zeros((m, 3))
        self.operator = (model.S.conj().T @ model.mass_correlation_operator[
            np.ix_(model.even_masks, model.even_masks)] @ model.S)
        self.target_metrics = [exact_metrics(c["target"], model) for c in candidates]

    def add(self, kets, covariances):
        kets = np.asarray(kets, np.complex128)
        if kets.ndim != 2 or kets.shape[1] != 128 or len(kets) == 0:
            raise ValueError("expected nonempty Gaussian batch (N,128)")
        if not np.all(np.isfinite(kets)) or not np.allclose(
                np.sum(abs(kets)**2, axis=1), 1., atol=1e-10, rtol=0):
            raise ValueError("Gaussian atoms must be normalized")
        invariants = atom_invariants(covariances, self.model.j_matrices)
        if not all(np.all(np.isfinite(v)) for v in invariants.values()):
            raise ValueError("nonfinite invariants")
        invariants["saturation_a"] = invariants["mass_a"]+invariants["eta"]
        invariants["saturation_b"] = invariants["mass_b"]+invariants["eta"]
        values = np.column_stack([invariants[name] for name in INVARIANTS])
        bins = invariant_bins(invariants["eta"], invariants["mass_a"])
        groups = eta_groups(invariants["eta"])
        halves = (self.count+np.arange(len(kets))) % 2
        overlap = np.einsum("ni,mi->nm", kets.conj(), self.u, optimize=False)
        absolute = abs(overlap)
        coefficients = self.d[None, :]*absolute**(2*self.orders[None, :]-2)*overlap
        mass = abs(coefficients)
        q = mass*absolute
        coordinates = np.einsum("ni,ij->nj", kets, self.model.S.conj(), optimize=False)
        norm2 = np.sum(abs(coordinates)**2, axis=1)
        self.raw_sum += np.einsum("nm,ni->mi", coefficients, kets, optimize=False)
        self.rb_sum += np.einsum("nm,nj->mj", coefficients, coordinates, optimize=False)
        self.rb_norm2_sum += np.einsum("nm,n->m", mass**2, norm2, optimize=False)
        self.mass_sum += mass.sum(axis=0)
        self.mass_sq_sum += (mass**2).sum(axis=0)
        self.q_sum += q.sum(axis=0)
        self.q_sq_sum += (q**2).sum(axis=0)
        self.max_mass = np.maximum(self.max_mass, mass.max(axis=0))
        self.haar_counts += np.bincount(bins, minlength=16)
        self.inv_sum += np.einsum("nm,nj->mj", mass, values, optimize=False)
        self.inv_w2_sum += np.einsum("nm,nj->mj", mass**2, values, optimize=False)
        self.inv_w2x2_sum += np.einsum("nm,nj->mj", mass**2, values**2, optimize=False)
        for i in range(len(self.candidates)):
            self.hist_sum[i] += np.bincount(bins, weights=mass[:, i], minlength=16)
            self.hist_sq_sum[i] += np.bincount(bins, weights=mass[:, i]**2, minlength=16)
        for group in range(3):
            selected = groups == group
            self.group_sum[:, group] += np.einsum("nm,nj->mj", coefficients[selected],
                                                coordinates[selected], optimize=False)
            self.group_norm2_sum[:, group] += np.einsum("nm,n->m", mass[selected]**2,
                                                       norm2[selected], optimize=False)
            self.group_mass_sum[:, group] += mass[selected].sum(axis=0)
            self.group_mass_sq_sum[:, group] += (mass[selected]**2).sum(axis=0)
            for half in range(2):
                chosen = selected & (halves == half)
                self.group_half_sum[:, half, group] += np.einsum("nm,nj->mj",
                    coefficients[chosen], coordinates[chosen], optimize=False)
        self.half_counts += np.bincount(halves, minlength=2)
        self.count += len(kets)

    def snapshot(self):
        fields = ("raw_sum", "rb_sum", "rb_norm2_sum", "mass_sum", "mass_sq_sum",
                  "q_sum", "q_sq_sum", "max_mass", "hist_sum", "hist_sq_sum", "haar_counts",
                  "inv_sum", "inv_w2_sum", "inv_w2x2_sum", "group_sum", "group_half_sum",
                  "group_norm2_sum", "group_mass_sum", "group_mass_sq_sum", "half_counts")
        result = {name: getattr(self, name).copy() for name in fields}
        result.update(completed_samples=np.asarray(self.count), witnesses=self.u.copy(),
                      orders=self.orders.copy(), basis=self.model.S.copy())
        return result

    def means(self):
        if self.count < 2 or np.any(self.half_counts == 0):
            raise ValueError("at least two samples required")
        n = self.count
        raw = self.raw_sum/n
        rb = np.einsum("ij,mj->mi", self.model.S, self.rb_sum/n, optimize=False)
        histogram = np.divide(self.hist_sum, self.mass_sum[:, None],
                              out=np.zeros_like(self.hist_sum), where=self.mass_sum[:, None] > 0)
        phi = self.group_sum/n
        halves = self.group_half_sum/self.half_counts[None, :, None, None]
        kernel = np.asarray([cross_kernel(h[0], h[1], self.operator) for h in halves])
        plugin = np.asarray([p.conj() @ self.operator @ p.T for p in phi])
        group_var = np.maximum(0., (self.group_norm2_sum-n*np.sum(abs(phi)**2, axis=2))/(n-1))
        group_rmse = np.sqrt(group_var/n)
        ess = np.divide(self.hist_sum**2, self.hist_sq_sum,
                        out=np.zeros_like(self.hist_sum), where=self.hist_sq_sum > 0)
        return {"raw_C": raw, "rb_C": rb, "histogram": histogram.reshape(-1, 4, 4),
                "haar_histogram": self.haar_counts.reshape(4, 4)/n,
                "histogram_ess": ess.reshape(-1, 4, 4), "group_phi": phi,
                "group_phi_halves": halves, "K_cross": kernel, "group_kernel_plugin": plugin,
                "group_rmse": group_rmse,
                "group_half_disagreement": np.linalg.norm(halves[:, 0]-halves[:, 1], axis=2)}

    def rows(self, config, previous_histogram=None):
        means = self.means()
        n = self.count
        m = len(self.candidates)
        total_events = int(config.get("planned_statistical_events", 5*m*len(config.get("seeds", [0]))))
        if total_events < 1:
            raise ValueError("planned_statistical_events must be positive")
        delta = float(config.get("total_failure_probability", config.get("failure_probability", .05)))/total_events
        if not 0 < delta < 1:
            raise ValueError("per-event failure probability must lie in (0,1)")
        reconstruction_tol = float(config.get("reconstruction_tolerance", .01))
        observable_tol = float(config.get("observable_tolerance", .05))
        rows = []
        for i, candidate in enumerate(self.candidates):
            target = candidate["target"]
            raw, rb = means["raw_C"][i], means["rb_C"][i]
            raw_error = float(np.linalg.norm(raw-target))
            rb_error = float(np.linalg.norm(rb-target))
            outside = float(np.linalg.norm(raw-rb))
            mass = self.mass_sum[i]/n
            q = self.q_sum[i]/n
            mass_var = max(0., (self.mass_sq_sum[i]-n*mass**2)/(n-1))
            q_var = max(0., (self.q_sq_sum[i]-n*q**2)/(n-1))
            rb_var = max(0., (self.rb_norm2_sum[i]-n*np.vdot(rb, rb).real)/(n-1))
            raw_var = max(0., (self.mass_sq_sum[i]-n*np.vdot(raw, raw).real)/(n-1))
            inv_mean = self.inv_sum[i]/self.mass_sum[i]
            delta_var = np.maximum(0., self.inv_w2x2_sum[i]
                -2*inv_mean*self.inv_w2_sum[i]+inv_mean**2*self.mass_sq_sum[i])
            inv_se = np.sqrt(n/(n-1)*delta_var)/self.mass_sum[i]
            norm = float(np.linalg.norm(rb))
            if norm <= 1e-15:
                metrics = {"mass_correlation": None, "wick_deviation": None}
            else:
                metrics = exact_metrics(rb/norm, self.model)
            reference = self.target_metrics[i]
            cq_cross = float(np.sum(means["K_cross"][i]).real)
            cq_error = abs(cq_cross-reference["mass_correlation"])
            w_error = abs(metrics["wick_deviation"]-reference["wick_deviation"]) if norm > 1e-15 else None
            highlighted = means["histogram"][i] >= float(config.get("highlighted_bin_min_mass", .05))
            bin_passed = bool(np.all(means["histogram_ess"][i][highlighted]
                                   >= float(config.get("highlighted_bin_min_ess", 100))))
            budget = formal_budget(self.dictionary, candidate["k"], float(np.linalg.norm(candidate["u"])),
                delta, float(config.get("statistical_radius_target", config.get("statistical_radius", .005))),
                int(config.get("final_samples", 65536)))
            radius = math.sqrt(budget["second_moment_bound"]/(delta*n))
            included_in_joint_budget = config.get("phase") in ("final", "validation") and not candidate["is_probe"]
            row = {"id": candidate["id"], "t": candidate.get("t"), "k": candidate["k"],
                "source": candidate["source"], "is_probe": bool(candidate["is_probe"]), "n": n,
                "witness_sha256": candidate["witness_sha256"], "target_sha256": _digest(target),
                "deterministic_verified": bool(candidate["deterministic_verified"]),
                "optimizer_converged": bool(candidate["optimizer_converged"]),
                "raw_residual": raw_error, "rb_residual": rb_error,
                "projected_residual": float(np.linalg.norm(self.model.S.conj().T @ (raw-target))),
                "outside_residual": outside,
                "pythagoras_error": abs(raw_error**2-rb_error**2-outside**2),
                "projection_commutation_error": float(np.linalg.norm(self.model.S.conj().T @ raw-self.rb_sum[i]/n)),
                "B": float(mass), "B_se": math.sqrt(mass_var/n), "q": float(q), "q_se": math.sqrt(q_var/n),
                "ess": float(self.mass_sum[i]**2/self.mass_sq_sum[i]),
                "max_weight_share": float(self.max_mass[i]/self.mass_sum[i]),
                "raw_rmse": math.sqrt(raw_var/n), "rb_rmse": math.sqrt(rb_var/n),
                "variance_reduction_fraction": float(1-rb_var/raw_var) if raw_var > 0 else None,
                "histogram_tv_previous": float(.5*np.abs(means["histogram"][i]-previous_histogram[i]).sum())
                    if previous_histogram is not None else None,
                "histogram_tv_haar": float(.5*np.abs(means["histogram"][i]-means["haar_histogram"]).sum()),
                "highlighted_bins_ess_passed": bin_passed,
                "highlighted_bin_minimum_ess": float(means["histogram_ess"][i][highlighted].min()) if np.any(highlighted) else None,
                "group_mass_fractions": (self.group_mass_sum[i]/self.mass_sum[i]).tolist(),
                "group_mass_ess": np.divide(self.group_mass_sum[i]**2, self.group_mass_sq_sum[i],
                    out=np.zeros(3), where=self.group_mass_sq_sum[i] > 0).tolist(),
                "group_rmse": means["group_rmse"][i].tolist(),
                "group_half_disagreement": means["group_half_disagreement"][i].tolist(),
                "group_norms": np.linalg.norm(means["group_phi"][i], axis=1).tolist(),
                "coherent_closure_residual": float(np.linalg.norm(np.sum(means["group_phi"][i], axis=0)
                    -self.model.S.conj().T @ target)),
                "group_additivity_error": float(np.linalg.norm(np.sum(means["group_phi"][i], axis=0)-self.rb_sum[i]/n)),
                "CQ_cross": cq_cross, "CQ_plugin": metrics["mass_correlation"],
                "CQ_reference": reference["mass_correlation"], "CQ_error": float(cq_error),
                "W_plugin": metrics["wick_deviation"], "W_reference": reference["wick_deviation"],
                "W_error": float(w_error) if w_error is not None else None,
                "reconstructed_norm2": norm**2,
                "local_reconstruction_passed": bool(candidate["optimizer_converged"]
                    and (candidate["deterministic_verified"] or rb_error <= reconstruction_tol)),
                "sampled_reconstruction_passed": rb_error <= reconstruction_tol,
                "local_coherent_passed": bool(candidate["optimizer_converged"] and rb_error <= reconstruction_tol and cq_error <= observable_tol
                    and w_error is not None and w_error <= observable_tol),
                "formal_budget": budget,
                "formal_envelopes": {"radius": radius, "B_lower": max(0., float(mass)-radius),
                    "B_upper": float(mass)+radius, "C_error_upper": rb_error+radius,
                    "each_group_vector_error_bound": radius,
                    "allocated_failure_probability_each": delta,
                    "allocated_events_this_candidate": 5 if included_in_joint_budget else 0,
                    "included_in_joint_budget": included_in_joint_budget,
                    "C_certified_within_tolerance": bool(included_in_joint_budget and rb_error+radius <= reconstruction_tol)},
                "histogram_formally_certified": False,
                "group_rmse_is_confidence_radius": False,
                "coherent_kernel_unbiased_for_unnormalized_CQ": True,
                "W_is_plugin_diagnostic": True,
            }
            row.update({"mean_"+name: float(inv_mean[j]) for j, name in enumerate(INVARIANTS)})
            row["invariant_se"] = {name: float(inv_se[j]) for j, name in enumerate(INVARIANTS)}
            if candidate["is_probe"]:
                analytic = gaussian_probe_reference(self.dictionary, candidate["k"])
                q_se = math.sqrt(max(0., analytic["q_second_moment"]-1)/n)
                b_se = math.sqrt(max(0., analytic["coefficient_second_moment"]-analytic["mass"]**2)/n)
                c_rmse = math.sqrt(max(0., analytic["coefficient_second_moment"]-1)/n)
                row.update(probe_reference=analytic, probe_q_se=q_se, probe_B_se=b_se, probe_C_rmse=c_rmse,
                    probe_passed=bool(abs(q-1.) <= 5*q_se+1e-12 and abs(mass-analytic["mass"]) <= 5*b_se+1e-12
                                      and raw_error <= 6*c_rmse+1e-12))
            rows.append(row)
        return rows, means


def run_statistics(model, candidates, config, seed, budget, checkpoint=None, phase=None):
    """Run one development or final stream after candidates have been frozen.

    Caller must complete all development work and freeze candidates, binning,
    thresholds and event accounting before starting any final stream.  The
    development and final calls never exchange samples or choose parameters.
    Checkpoints contain sufficient sums and RNG state for auditing/resumption.
    """
    phase = str(phase or config.get("phase", "development"))
    phase = "final" if phase == "validation" else phase
    if phase not in ("development", "final"):
        raise ValueError("phase must be development or final")
    sizes = ([int(config.get("final_samples", 65536))] if phase == "final"
             else [int(n) for n in config.get("sample_sizes", [4096, 16384, 65536])])
    if not sizes or sizes != sorted(set(sizes)) or sizes[0] < 2:
        raise ValueError("sample sizes must increase and be >= 2")
    batch_size = int(config.get("batch_size", 256))
    if batch_size < 1 or batch_size > 256 or sizes[-1] > 65536:
        raise ValueError("pilot caps are batch <=256 and samples <=65536")
    budget.check()
    prepared, probe_metadata = _prepare_candidates(model, candidates, bool(config.get("include_probes", True)))
    dictionary = Dictionary("full")
    labels = [int(seed), 910, 2 if phase == "development" else 4]
    rng = np.random.default_rng(np.random.SeedSequence(labels))
    accumulator = StructureAccumulator(model, prepared)
    report = {"status": "running", "phase": phase, "seed": int(seed), "rng_seed_sequence": labels,
        "planned_sample_sizes": sizes, "candidate_ids": [c["id"] for c in prepared], "rows": [],
        "probe_metadata": probe_metadata, "candidate_count": len(prepared),
        "bin_edges": [0., .25, .5, .75, 1.], "eta_group_upper_edges": [.25, .75, 1.],
        "coherent_group_halves": "alternating iid samples, split independent of samples and coefficients",
        "group_coordinates_basis": "model.S; unnormalized projected Gaussian orbit averages",
        "histogram_has_formal_confidence_guarantee": False,
        "formal_bound_kind": "second-moment Markov/Chebyshev and union bound, conditional on ideal Haar and floating-point operators",
        "strict_interval_certified": False,
        "probe_failure_budget": "diagnostic only, excluded from formal canonical-event accounting"}
    arrays = {"candidate_ids": np.asarray([c["id"] for c in prepared]),
        "targets": np.asarray([c["target"] for c in prepared]),
        "coherent_operator": accumulator.operator.copy(),
        "is_probe": np.asarray([c["is_probe"] for c in prepared], bool)}
    previous_histogram = None
    safe_state = (accumulator.snapshot(), copy.deepcopy(rng.bit_generator.state), 0)

    def save_checkpoint():
        nonlocal safe_state
        # A stage alarm may interrupt a batch between in-place sum updates.
        # Only a completed checkpoint is eligible for a saved partial result.
        snapshot = accumulator.snapshot()
        safe_state = (snapshot, copy.deepcopy(rng.bit_generator.state), len(report["rows"]))
        arrays.update(snapshot)
        report.update(completed_samples=accumulator.count, rng_state=rng.bit_generator.state, usage=budget.usage())
        if checkpoint:
            checkpoint(report, arrays)

    try:
        for size in sizes:
            while accumulator.count < size:
                budget.check()
                count = min(batch_size, size-accumulator.count)
                batch = dictionary.sample(count, rng)
                accumulator.add(batch.kets, batch.covariances)
                if accumulator.count % 4096 == 0 and accumulator.count != size:
                    save_checkpoint()
            budget.check()
            rows, means = accumulator.rows(dict(config, phase=phase), previous_histogram)
            for row in rows:
                row.update(phase=phase, seed=int(seed))
            report["rows"].extend(rows)
            for name, value in means.items():
                arrays[f"n{size}_{name}"] = value.copy()
            previous_histogram = means["histogram"].copy()
            save_checkpoint()
        report["status"] = "complete"
    except ResourceLimit as error:
        report.update(status="resource_limit", error=str(error))
        snapshot, safe_rng, row_count = safe_state
        for name, value in snapshot.items():
            if name == "completed_samples":
                accumulator.count = int(value)
            elif name not in ("witnesses", "orders", "basis"):
                setattr(accumulator, name, value.copy())
        rng.bit_generator.state = safe_rng
        report["rows"] = report["rows"][:row_count]
        for name in list(arrays):
            if name.startswith("n") and name.split("_", 1)[0][1:].isdigit():
                if int(name.split("_", 1)[0][1:]) > accumulator.count:
                    del arrays[name]
        report["partial_state_rollback"] = "last completed consistent checkpoint; in-flight batch discarded"
    save_checkpoint()
    final_probes = [r for r in report["rows"] if r["is_probe"] and r["n"] == sizes[-1]]
    report["probe_status"] = ("not_requested" if not config.get("include_probes", True) else
        "incomplete" if len(final_probes) != 2 else
        "passed" if all(r["probe_passed"] for r in final_probes) else "failed")
    return report, arrays
