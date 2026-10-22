"""One-shot independent validation; never reroll until a certificate passes."""

from dataclasses import dataclass, asdict
import hashlib
import math
import numpy as np
from .moments import MomentAccumulator, batch_moments, scalar_moment
from .solver import ResourceLimitError
from .theory import log_dimension, log_second_moment_bound


@dataclass
class ValidationReport:
    status: str
    samples: int
    requested_samples: int | None
    delta: float | None
    tolerance: float
    candidate_sha256: str
    residual_estimate: float | None = None
    statistical_radius: float | None = None
    upper_bound: float | None = None
    original_pdf_radius: float | None = None
    mass_estimate: float | None = None
    mass_standard_error: float | None = None
    q_target_estimate: float | None = None
    second_moment_bound: float | None = None
    reconstruction: np.ndarray | None = None
    error: str | None = None

    @property
    def passed(self):
        return self.status in ("certified", "exact")

    def summary(self):
        result = asdict(self)
        result.pop("reconstruction")
        result["passed"] = self.passed
        result["mass_standard_error_is_confidence_interval"] = False
        return result


def validate(psi, u, k, sampler, budget, *, delta=.025, tolerance=.05,
             margin=.025, batch_size=8192, max_samples=12_000_000,
             k1_mass_samples=65536):
    """Caller must freeze u before using this independent sampler.

    For k>1, N is chosen from u and the analytic second-moment bound only.
    The target-q and mass estimates share this held-out batch; they are not
    additional confidence claims and never influence N or candidate selection.
    """
    psi, u = sampler.basis.vector(psi, nonzero=True), sampler.basis.vector(u)
    if (not 0 < delta < 1 or not 0 < margin < tolerance
            or not 1 <= batch_size <= 8192 or not 1 <= max_samples <= 12_000_000
            or not 1 <= k1_mass_samples <= max_samples):
        raise ValueError("Invalid validation precision or resource bounds")
    u.flags.writeable = False
    digest = hashlib.sha256(u.tobytes()).hexdigest()
    norm = float(np.linalg.norm(u))
    report = ValidationReport("pending", 0, None, None if k == 1 else delta,
                              tolerance, digest)
    if norm == 0:
        residual = float(np.linalg.norm(psi))
        report.status = "exact" if residual <= tolerance else "not_certified"
        report.residual_estimate = report.upper_bound = residual
        report.statistical_radius = report.original_pdf_radius = 0.
        report.mass_estimate = 0.
        report.reconstruction = u.copy()
        report.requested_samples = 0
        return report
    log_v = log_second_moment_bound(sampler.basis.n, k, norm)
    if k == 1:
        requested = k1_mass_samples
    else:
        log_requested = log_v - 2*math.log(margin) - math.log(delta)
        if log_requested > math.log(max_samples):
            report.status = "sample_budget_exceeded"
            report.requested_samples = math.ceil(math.exp(log_requested)) if log_requested < 700 else None
            report.error = "Required independent sample count exceeds per-order budget; no draws used"
            return report
        requested = max(1, math.ceil(math.exp(log_requested)))
    report.requested_samples = requested
    accumulator = MomentAccumulator(len(u))
    q_target_sum = 0.
    q_target_correction = 0.
    try:
        budget.check()
        while accumulator.count < requested:
            budget.check()
            g = sampler.sample(min(batch_size, requested-accumulator.count))
            accumulator.add(batch_moments(u, k, g, n=sampler.basis.n))
            if k != 1:
                weighted_q = scalar_moment(psi, 2*k, g,
                                           log_prefactor=log_dimension(sampler.basis.n, k))*len(g)
                value = weighted_q-q_target_correction
                updated = q_target_sum+value
                q_target_correction = (updated-q_target_sum)-value
                q_target_sum = updated
        budget.check()
    except (ResourceLimitError, FloatingPointError, OverflowError) as exc:
        report.status = "resource_limit" if isinstance(exc, ResourceLimitError) else "numerical_error"
        report.samples = accumulator.count
        report.error = str(exc)
        # Partial, resource-stopped data do not receive a fixed-N certificate.
        return report
    estimate = accumulator.result()
    report.samples = requested
    report.mass_estimate = estimate.mass
    report.mass_standard_error = estimate.mass_standard_error
    if k == 1:
        report.reconstruction = u.copy()
        report.q_target_estimate = float(np.vdot(psi, psi).real)
        report.statistical_radius = report.original_pdf_radius = 0.
    else:
        report.reconstruction = estimate.C
        report.q_target_estimate = q_target_sum/requested
        report.second_moment_bound = math.exp(log_v)
        report.statistical_radius = math.exp(.5*(log_v-math.log(requested)-math.log(delta)))
        report.original_pdf_radius = math.exp(log_dimension(sampler.basis.n, k)
            +(2*k-1)*math.log(norm)-.5*(math.log(requested)+math.log(delta)))
    report.residual_estimate = float(np.linalg.norm(psi-report.reconstruction))
    report.upper_bound = report.residual_estimate+report.statistical_radius
    report.status = ("exact" if k == 1 else "certified") if report.upper_bound <= tolerance else "not_certified"
    return report
