"""Resource-bounded finite-order Gaussian extent estimation.

The schedule is fixed before training. Each frozen witness gets at most one
held-out validation; all planned stochastic orders share a fixed failure budget.
Sparse schedules bound extent but do not evaluate the full Xi_K hierarchy.
"""

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path

import numpy as np

from .basis import FockBasis
from .certificates import ExtentCertificate, extent_certificate
from .sampling import HaarFourModeSampler
from .solver import (ResourceBudget, ResourceLimitError, SolveResult,
                     peak_memory_mib, projected_sgd)
from .validation import ExtentValidationReport, validate_extent_candidate


@dataclass(frozen=True)
class ExtentConfig:
    orders: tuple = (1, 2, 4)
    seed: int = 20261022
    iterations: int = 8000
    training_batch: int = 2048
    validation_batch: int = 8192
    diagnostic_samples: int = 16384
    solver_mode: str = "continuation"
    relative_tolerance: float = .1
    total_failure_probability: float = .05
    validation_margin: float = .025
    max_validation_samples: int = 12_000_000
    max_seconds: float = 180.
    max_memory_mib: float = 512.

    def __post_init__(self):
        orders = tuple(self.orders)
        if (not orders or orders != tuple(sorted(set(orders)))
                or any(type(k) is not int or k not in (1, 2, 4) for k in orders)):
            raise ValueError("v1 orders must be an increasing subset of (1,2,4)")
        object.__setattr__(self, "orders", orders)
        bounds = {
            "iterations": (2, 8000), "training_batch": (1, 8192),
            "validation_batch": (1, 8192), "diagnostic_samples": (1, 65536),
            "max_validation_samples": (1, 12_000_000),
        }
        for name, (lower, upper) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.solver_mode not in ("continuation", "theory"):
            raise ValueError("Unknown solver mode")
        for name in ("relative_tolerance", "validation_margin", "max_seconds", "max_memory_mib"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (not np.isfinite(self.total_failure_probability)
                or not 0 < self.total_failure_probability < 1):
            raise ValueError("Failure probability must be between zero and one")
        if self.max_seconds > 180 or self.max_memory_mib > 512:
            raise ValueError("v1 resource limits cannot exceed 180 seconds / 512 MiB")


@dataclass
class ExtentStageResult:
    k: int
    solution: SolveResult
    rng_spawn_keys: dict
    validation: ExtentValidationReport | None = None
    certificate: ExtentCertificate | None = None
    applied: bool = False
    error: str | None = None


@dataclass
class ExtentResult:
    psi: np.ndarray
    basis: FockBasis
    config: ExtentConfig
    lower_bound: float
    upper_bound: float
    confidence: float = 1.
    interval_certified: bool = True
    target_reached: bool = False
    stop_reason: str = "running"
    error: str | None = None
    stages: list = field(default_factory=list)
    elapsed_seconds: float = 0.
    peak_memory_mib: float = 0.

    @property
    def relative_gap(self):
        return self.upper_bound/self.lower_bound-1

    @property
    def witnesses(self):
        return {stage.k: stage.solution.u.copy() for stage in self.stages}

    @property
    def history(self):
        return [dict(k=stage.k, **row)
                for stage in self.stages for row in stage.solution.history]

    def summary(self):
        return {
            "schema_version": 1, "task": "gaussian_extent",
            "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
            "relative_gap": self.relative_gap, "confidence": self.confidence,
            "interval_certified": self.interval_certified,
            "target_reached": self.target_reached, "stop_reason": self.stop_reason,
            "error": self.error, "config": asdict(self.config),
            "basis": {"n": self.basis.n, "parity": self.basis.parity,
                      "occupation_bitmasks": self.basis.occupations,
                      "convention": "mode j is bit j; creation operators in ascending j"},
            "input_norm": float(np.linalg.norm(self.psi)),
            "input_automatically_normalized": False,
            "baseline_bounds": "||psi||_2^2 <= extent(psi) <= ||psi||_1^2",
            "hierarchy": "per-order bounds; sparse orders do not evaluate full Xi_K",
            "confidence_assumptions": "independent ideal Haar sampling; floating-point rounding is not interval-certified",
            "confidence_allocation": "union bound over all planned stochastic orders, including unvisited orders",
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mib": self.peak_memory_mib,
            "numpy_version": np.__version__, "rng_algorithm": "PCG64",
            "stages": [{
                "k": stage.k, "solver_status": stage.solution.status,
                "solver_error": stage.solution.error, "iterations": stage.solution.iterations,
                "search_radius": stage.solution.radius,
                "u_norm": float(np.linalg.norm(stage.solution.u)),
                "rng_spawn_keys": stage.rng_spawn_keys,
                "validation": stage.validation.summary() if stage.validation else None,
                "certificate": asdict(stage.certificate) if stage.certificate else None,
                "applied": stage.applied, "error": stage.error,
                "history": stage.solution.history,
            } for stage in self.stages],
        }

    def save(self, directory):
        """Save a new run; refuse to overwrite previously recorded evidence."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        names = ("summary.json", "vectors.npz", "history.csv", "intervals.csv")
        if any((directory/name).exists() for name in names):
            raise FileExistsError("Extent result files already exist; choose a new output directory")
        (directory/"summary.json").write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False, allow_nan=False)+"\n",
            encoding="utf-8")
        arrays = {"psi": self.psi}
        for stage in self.stages:
            arrays[f"u_k{stage.k}"] = stage.solution.u
            if stage.validation and stage.validation.reconstruction is not None:
                arrays[f"reconstruction_k{stage.k}"] = stage.validation.reconstruction
        np.savez_compressed(directory/"vectors.npz", **arrays)
        rows = self.history
        with (directory/"history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["k", "iteration"])
            writer.writeheader()
            writer.writerows(rows)
        rows = []
        for stage in self.stages:
            certificate, validation = stage.certificate, stage.validation
            rows.append({
                "k": stage.k, "applied": stage.applied,
                "validation_status": validation.status if validation else "not_run",
                "validation_samples": validation.samples if validation else 0,
                "t_lower": certificate.t_lower if certificate else None,
                "t_upper": certificate.t_upper if certificate else None,
                "extent_lower": certificate.lower_bound if certificate else None,
                "extent_upper": certificate.upper_bound if certificate else None,
                "q_lower": validation.q_lower if validation else None,
                "q_upper": validation.q_upper if validation else None,
                "residual_bound": validation.residual_bound if validation else None,
            })
        with (directory/"intervals.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["k", "applied"])
            writer.writeheader()
            writer.writerows(rows)


def _intersect_bounds(lower, upper, certificate):
    new_lower = max(lower, certificate.lower_bound)
    new_upper = min(upper, certificate.upper_bound)
    if new_lower > new_upper:
        # Only widen an inversion at floating-point roundoff scale. A genuine
        # conflict can be a failed statistical event and must not be accepted.
        tolerance = 64*np.finfo(float).eps*max(new_lower, new_upper)
        if new_lower-new_upper > tolerance:
            raise ValueError("Extent certificate conflicts with previously valid bounds")
        new_lower, new_upper = new_upper, new_lower
    return new_lower, new_upper


def estimate_extent(psi, basis=None, config=None, *, progress=None, budget=None):
    """Return a certified interval, which need not reach the requested precision.

    Accepts unnormalized nonzero four-mode even-parity Fock vectors. No magic
    state analytic data enters this routine. Resource failure preserves earlier
    intervals, including the deterministic Fock bounds available before sampling.
    """
    basis, config = basis or FockBasis(), config or ExtentConfig()
    basis.require_four_mode_even()
    psi = basis.vector(psi, nonzero=True)
    with np.errstate(over="ignore", under="ignore"):
        lower = float(np.vdot(psi, psi).real)
        upper = float(np.sum(np.abs(psi))**2)
    if not np.isfinite(upper) or not np.isfinite(lower) or lower <= 0:
        raise ValueError("Input squared norms must be representable in float64; rescale psi")
    upper = max(lower, upper)
    budget = budget or ResourceBudget(config.max_seconds, config.max_memory_mib)
    if (not np.isfinite(budget.max_seconds) or not np.isfinite(budget.max_memory_mib)
            or budget.max_seconds <= 0 or budget.max_memory_mib <= 0
            or budget.max_seconds > config.max_seconds
            or budget.max_memory_mib > config.max_memory_mib):
        raise ValueError("An external budget cannot relax the configured resource limits")
    result = ExtentResult(psi, basis, config, lower, upper)
    count = sum(k != 1 for k in config.orders)
    delta = config.total_failure_probability/max(1, count)
    roots = np.random.SeedSequence(config.seed).spawn(len(config.orders))
    previous = psi.copy()
    try:
        budget.check()
        if result.relative_gap <= config.relative_tolerance:
            result.target_reached = True
            result.stop_reason = "target_reached"
        else:
            for k, root in zip(config.orders, roots):
                budget.check()
                streams = root.spawn(3)
                samplers = [HaarFourModeSampler(np.random.Generator(np.random.PCG64(s)), basis)
                            for s in streams]
                solution = projected_sgd(
                    psi, k, samplers[0], samplers[1], budget, initial=previous,
                    iterations=config.iterations, batch_size=config.training_batch,
                    mode=config.solver_mode, diagnostic_samples=config.diagnostic_samples,
                    callback=progress)
                stage = ExtentStageResult(k, solution, {
                    name: list(s.spawn_key) for name, s in
                    zip(("training", "diagnostic", "validation"), streams)})
                result.stages.append(stage)
                if solution.status not in ("completed", "exact"):
                    result.stop_reason, result.error = solution.status, solution.error
                    break
                stage.validation = validate_extent_candidate(
                    psi, solution.u, k, samplers[2], budget, delta=delta,
                    margin=config.validation_margin, batch_size=config.validation_batch,
                    max_samples=config.max_validation_samples)
                report = stage.validation
                if report.status != "certified":
                    result.stop_reason, result.error = report.status, report.error
                    break
                try:
                    stage.certificate = extent_certificate(
                        psi, solution.u, k, n=basis.n, q_lower=report.q_lower,
                        q_upper=report.q_upper, residual_bound=report.residual_bound)
                    lower, upper = _intersect_bounds(
                        result.lower_bound, result.upper_bound, stage.certificate)
                except ValueError as exc:
                    stage.error = str(exc)
                    result.stop_reason, result.error = "inconsistent_bounds", str(exc)
                    break
                result.lower_bound, result.upper_bound = lower, upper
                stage.applied = True
                if report.delta is not None and report.delta > 0:
                    # The guarantee uses the whole preallocated budget, not a
                    # post-selected confidence based on the observed stop time.
                    result.confidence = 1-config.total_failure_probability
                previous = solution.u.copy()
                if result.relative_gap <= config.relative_tolerance:
                    result.target_reached = True
                    result.stop_reason = "target_reached"
                    break
            if result.stop_reason == "running":
                result.stop_reason = "orders_exhausted"
    except (ResourceLimitError, FloatingPointError, OverflowError) as exc:
        result.stop_reason = "resource_limit" if isinstance(exc, ResourceLimitError) else "numerical_error"
        result.error = str(exc)
    result.elapsed_seconds = budget.elapsed
    result.peak_memory_mib = peak_memory_mib()
    return result
