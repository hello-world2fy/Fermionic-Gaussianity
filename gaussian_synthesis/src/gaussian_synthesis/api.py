"""Public configuration, bounded execution and reproducible result serialization."""

from dataclasses import dataclass, field, asdict
import csv
import json
from pathlib import Path
import numpy as np
from .basis import FockBasis
from .measure import GaussianCoefficientMeasure
from .sampling import HaarFourModeSampler
from .solver import ResourceBudget, ResourceLimitError, SolveResult, peak_memory_mib, projected_sgd
from .theory import dimension, gamma_factor
from .validation import ValidationReport, validate


@dataclass(frozen=True)
class SynthesisConfig:
    orders: tuple = (1, 2, 4)
    seed: int = 20261022
    iterations: int = 8000
    training_batch: int = 2048
    validation_batch: int = 8192
    diagnostic_samples: int = 16384
    k1_mass_samples: int = 65536
    solver_mode: str = "continuation"
    tolerance: float = .05
    total_failure_probability: float = .05
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
            "k1_mass_samples": (1, 12_000_000), "max_validation_samples": (1, 12_000_000),
        }
        for name, (lower, upper) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if self.k1_mass_samples > self.max_validation_samples:
            raise ValueError("k1_mass_samples exceeds the validation sample limit")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.solver_mode not in ("continuation", "theory"):
            raise ValueError("Unknown solver mode")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        if not 0 < self.total_failure_probability < 1:
            raise ValueError("Failure probability must be between zero and one")
        if not 0 < self.max_seconds <= 180 or not 0 < self.max_memory_mib <= 512:
            raise ValueError("v1 resource limits cannot exceed 180 seconds / 512 MiB")


@dataclass
class StageResult:
    k: int
    solution: SolveResult
    measure: GaussianCoefficientMeasure
    validation: ValidationReport | None
    rng_spawn_keys: dict


@dataclass
class SynthesisResult:
    psi: np.ndarray
    basis: FockBasis
    config: SynthesisConfig
    stages: list = field(default_factory=list)
    status: str = "running"
    elapsed_seconds: float = 0.
    peak_memory_mib: float = 0.
    error: str | None = None

    def summary(self):
        return {
            "schema_version": 1, "status": self.status, "error": self.error,
            "basis": {"n": self.basis.n, "parity": self.basis.parity,
                      "occupation_bitmasks": self.basis.occupations,
                      "convention": "mode j is bit j; creation operators in ascending j"},
            "config": asdict(self.config), "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mib": self.peak_memory_mib,
            "numpy_version": np.__version__, "rng_algorithm": "PCG64",
            "joint_reconstruction_confidence": (1-self.config.total_failure_probability
                if self.status == "certified" else None),
            "confidence_assumptions": "independent ideal Haar sampling; floating-point rounding is not interval-certified",
            "stages": [{
                "k": stage.k, "d_nk": dimension(self.basis.n, stage.k),
                "finite_order_gamma": gamma_factor(self.basis.n, stage.k),
                "solver_status": stage.solution.status, "solver_error": stage.solution.error,
                "iterations": stage.solution.iterations, "search_radius": stage.solution.radius,
                "u_norm": float(np.linalg.norm(stage.solution.u)),
                "rng_spawn_keys": stage.rng_spawn_keys,
                "reference_measure": stage.measure.reference_measure,
                "reconstructed_state": "C_k(u); close to psi only to the reported certified precision",
                "validation": stage.validation.summary() if stage.validation else None,
                "history": stage.solution.history,
            } for stage in self.stages],
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/"summary.json").write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False,
                                                       allow_nan=False)+"\n", encoding="utf-8")
        arrays = {"psi": self.psi}
        for stage in self.stages:
            arrays[f"u_k{stage.k}"] = stage.solution.u
            if stage.validation and stage.validation.reconstruction is not None:
                arrays[f"reconstruction_k{stage.k}"] = stage.validation.reconstruction
        np.savez_compressed(directory/"vectors.npz", **arrays)
        rows = [dict(k=stage.k, **row) for stage in self.stages for row in stage.solution.history]
        with (directory/"history.csv").open("w", newline="", encoding="utf-8") as handle:
            names = list(rows[0]) if rows else ["k", "iteration"]
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)


def synthesize(psi, basis=None, config=None, *, progress=None, budget=None):
    """Full-space optimization, then exactly one held-out validation per order.

    A caller-supplied budget lets a runner include process startup and output in
    the same deadline. No analytic magic-state information enters this API.
    """
    basis, config = basis or FockBasis(), config or SynthesisConfig()
    basis.require_four_mode_even()
    psi = basis.vector(psi, nonzero=True)
    budget = budget or ResourceBudget(config.max_seconds, config.max_memory_mib)
    if budget.max_seconds > config.max_seconds or budget.max_memory_mib > config.max_memory_mib:
        raise ValueError("An external budget cannot relax the configured resource limits")
    result = SynthesisResult(psi, basis, config)
    stochastic_count = sum(k != 1 for k in config.orders)
    delta = config.total_failure_probability/max(1, stochastic_count)
    roots = np.random.SeedSequence(config.seed).spawn(len(config.orders))
    previous = psi
    try:
        for k, root in zip(config.orders, roots):
            budget.check()
            streams = root.spawn(3)
            samplers = [HaarFourModeSampler(np.random.Generator(np.random.PCG64(s)), basis) for s in streams]
            solution = projected_sgd(psi, k, samplers[0], samplers[1], budget,
                initial=previous, iterations=config.iterations, batch_size=config.training_batch,
                mode=config.solver_mode, diagnostic_samples=config.diagnostic_samples, callback=progress)
            stage = StageResult(k, solution, GaussianCoefficientMeasure(basis, k, solution.u), None,
                {name: list(s.spawn_key) for name, s in zip(("training", "diagnostic", "validation"), streams)})
            result.stages.append(stage)
            if solution.status not in ("completed", "exact"):
                result.status, result.error = solution.status, solution.error
                break
            # Neither the analytic reference nor held-out data select this u.
            stage.validation = validate(psi, solution.u, k, samplers[2], budget,
                delta=delta, tolerance=config.tolerance, margin=config.tolerance/2,
                batch_size=config.validation_batch, max_samples=config.max_validation_samples,
                k1_mass_samples=config.k1_mass_samples)
            previous = solution.u.copy()
            if stage.validation.status in ("resource_limit", "numerical_error"):
                result.status, result.error = stage.validation.status, stage.validation.error
                break
        if result.status == "running":
            result.status = "certified" if all(s.validation and s.validation.passed for s in result.stages) else "not_certified"
    except (ResourceLimitError, FloatingPointError, OverflowError) as exc:
        result.status = "resource_limit" if isinstance(exc, ResourceLimitError) else "numerical_error"
        result.error = str(exc)
    result.elapsed_seconds = budget.elapsed
    result.peak_memory_mib = peak_memory_mib()
    return result
