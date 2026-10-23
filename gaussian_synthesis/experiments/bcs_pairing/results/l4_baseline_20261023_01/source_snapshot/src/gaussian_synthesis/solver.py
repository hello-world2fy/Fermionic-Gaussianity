"""Projected SGD and cooperative resource monitoring; no analytic magic solution."""

from dataclasses import dataclass, field
import math
import resource
import sys
import time
import numpy as np
from .moments import estimate_moments
from .objective import ConvexObjective
from .theory import log_dimension, search_radius


class ResourceLimitError(RuntimeError):
    pass


def peak_memory_mib():
    # macOS ru_maxrss is bytes; Linux reports KiB.
    factor = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor/2**20


@dataclass
class ResourceBudget:
    max_seconds: float = 180.
    max_memory_mib: float = 512.
    started: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self):
        return time.monotonic()-self.started

    def check(self):
        if self.elapsed >= self.max_seconds:
            raise ResourceLimitError(f"Wall-clock budget exceeded ({self.max_seconds:g} s)")
        memory = peak_memory_mib()
        if memory > self.max_memory_mib:
            raise ResourceLimitError(f"Peak RSS {memory:.1f} MiB exceeds budget")


@dataclass
class SolveResult:
    u: np.ndarray
    k: int
    status: str
    iterations: int
    radius: float
    history: list
    error: str | None = None


def project_ball(u, radius):
    norm = np.linalg.norm(u)
    if not np.isfinite(norm):
        raise FloatingPointError("Nonfinite iterate")
    return u*(radius/norm) if norm > radius else u.copy()


def projected_sgd(psi, k, sampler, diagnostic_sampler, budget, *,
                  initial=None, iterations=8000, batch_size=2048,
                  mode="continuation", diagnostic_samples=16384, callback=None):
    radius = search_radius(sampler.basis.n, k, float(np.linalg.norm(psi)))
    if k == 1:
        budget.check()
        return SolveResult(psi.copy(), k, "exact", 0, radius, [])
    objective = ConvexObjective(psi, k, sampler.basis.n)
    if mode == "theory":
        u = np.zeros_like(psi)
        log_g_atom = log_dimension(sampler.basis.n, k)+(2*k-1)*math.log(radius)
        G = float(np.exp(np.logaddexp(log_g_atom, math.log(np.linalg.norm(psi)))))
        theory_step = radius/(G*math.sqrt(iterations))
    elif mode == "continuation":
        u = project_ball(psi if initial is None else initial, radius)
    else:
        raise ValueError("solver_mode must be continuation or theory")
    summed = np.zeros_like(u)
    average_count = 0
    history = []
    completed = 0
    status, error = "completed", None
    try:
        for t in range(iterations):
            budget.check()
            # The theorem averages pre-update iterates from zero. The practical
            # mode averages the last half of post-update iterates instead.
            if mode == "theory":
                summed += u
                average_count += 1
            g = sampler.sample(batch_size)
            objective_value, gradient = objective.value_gradient(u, g)
            step = theory_step if mode == "theory" else .1*(1+t/200)**(-.6)/(2*k-1)
            u = project_ball(u-step*gradient, radius)
            completed = t+1
            if mode != "theory" and t >= iterations//2:
                summed += u
                average_count += 1
            if t == 0 or completed % 1000 == 0 or completed == iterations:
                candidate = summed/average_count if average_count else u
                diagnostic = estimate_moments(candidate, k, diagnostic_sampler,
                                              diagnostic_samples, check=budget.check)
                row = {
                    "iteration": completed, "step_size": step,
                    "training_objective_estimate": objective_value,
                    "diagnostic_objective_estimate": float(diagnostic.q/(2*k)-np.vdot(candidate, psi).real),
                    "diagnostic_residual_estimate": float(np.linalg.norm(psi-diagnostic.C)),
                    "candidate_norm": float(np.linalg.norm(candidate)),
                    "elapsed_seconds": budget.elapsed,
                }
                history.append(row)
                if callback:
                    callback(k, row)
    except (ResourceLimitError, FloatingPointError, OverflowError) as exc:
        status, error = "resource_limit" if isinstance(exc, ResourceLimitError) else "numerical_error", str(exc)
    return SolveResult((summed/average_count if average_count else u).copy(),
                       k, status, completed, radius, history, error)
