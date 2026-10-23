"""One-shot independent validation; never reroll until a certificate passes."""

from dataclasses import dataclass, asdict
import hashlib
import math
from numbers import Real
import numpy as np
from .moments import MomentAccumulator, batch_moments, scalar_moment
from .solver import ResourceLimitError
from .theory import check_order, log_dimension, log_second_moment_bound


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


@dataclass
class ExtentValidationReport:
    """A joint q/residual integration certificate for one frozen candidate.

    ``certified`` describes the integration bounds, not attainment of a target
    extent precision. The reconstruction array is stored separately from JSON.
    """

    status: str
    samples: int
    requested_samples: int | None
    delta: float | None
    candidate_sha256: str
    q_estimate: float | None = None
    q_lower: float | None = None
    q_upper: float | None = None
    reconstruction: np.ndarray | None = None
    residual_estimate: float | None = None
    statistical_radius: float | None = None
    residual_bound: float | None = None
    error: str | None = None

    def summary(self):
        result = asdict(self)
        result.pop("reconstruction")
        return result


def validate_extent_candidate(psi, u, k, sampler, budget, *, delta=.025,
                              margin=.025, batch_size=8192,
                              max_samples=12_000_000):
    """Validate once with a sampler independent of candidate selection.

    The sample count is fixed before any draws, using the frame second-moment
    bound. Euler's identity q_k(u) = Re <u, C_k(u)> converts the same confidence
    ball into a q interval, without spending another failure probability.
    Repeated validation/selection must be accounted for by the caller's joint
    probability allocation. Partial data receive diagnostics but no certificate.
    """
    check_order(k)
    psi = sampler.basis.vector(psi, nonzero=True)
    u = sampler.basis.vector(u)
    for name, value in (("delta", delta), ("margin", margin)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite")
    if delta >= 1:
        raise ValueError("delta must be less than one")
    for name, value, cap in (("batch_size", batch_size, 8192),
                             ("max_samples", max_samples, 12_000_000)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 1 <= value <= cap):
            raise ValueError(f"{name} must be an integer between 1 and {cap}")
    # A private immutable copy prevents caller changes during validation.
    u.flags.writeable = False
    norm = float(np.linalg.norm(u))
    exact = k == 1 or norm == 0
    report = ExtentValidationReport(
        "pending", 0, None, None if exact else float(delta),
        hashlib.sha256(u.tobytes()).hexdigest())
    total = np.zeros_like(u)
    correction = np.zeros_like(u)

    def diagnostics():
        if report.samples:
            reconstruction = total / report.samples
            if np.all(np.isfinite(reconstruction)):
                report.reconstruction = reconstruction.copy()
                with np.errstate(over="ignore", invalid="ignore"):
                    q = float(np.vdot(u, reconstruction).real)
                    residual = float(np.linalg.norm(psi - reconstruction))
                if math.isfinite(q):
                    report.q_estimate = q
                if math.isfinite(residual):
                    report.residual_estimate = residual

    try:
        budget.check()
        if norm == 0 and np.any(u != 0):
            raise FloatingPointError("Candidate norm underflows float64 range")
        if exact:
            reconstruction = u.copy() if k == 1 else np.zeros_like(u)
            with np.errstate(over="raise", invalid="raise"):
                q = float(np.vdot(u, u).real) if k == 1 else 0.
                residual = float(np.linalg.norm(psi - reconstruction))
            if not math.isfinite(q) or not math.isfinite(residual):
                raise FloatingPointError("Exact validation exceeds float64 range")
            report.status = "certified"
            report.requested_samples = 0
            report.q_estimate = report.q_lower = report.q_upper = q
            report.reconstruction = reconstruction
            report.residual_estimate = report.residual_bound = residual
            report.statistical_radius = 0.
            return report

        log_norm = math.log(norm)
        log_d = log_dimension(sampler.basis.n, k)
        # These deterministic bounds follow from Q_k <= I and Jensen + frame.
        deterministic_upper = math.exp(2*k*log_norm)
        deterministic_lower = math.exp(
            log_d + 2*k*log_norm - k*math.log(len(u)))
        if (not math.isfinite(deterministic_upper)
                or deterministic_upper == 0
                or not math.isfinite(deterministic_lower)):
            raise FloatingPointError("Moment bounds exceed float64 range")
        log_v = log_second_moment_bound(sampler.basis.n, k, norm)
        log_requested = log_v - 2*math.log(margin) - math.log(delta)
        if not math.isfinite(log_v) or not math.isfinite(log_requested):
            raise FloatingPointError("Validation sample bound is nonfinite")
        if log_requested > math.log(max_samples):
            report.status = "sample_budget_exceeded"
            report.requested_samples = (
                math.ceil(math.exp(log_requested)) if log_requested < 700 else None)
            report.error = (
                "Required independent sample count exceeds per-order budget; no draws used")
            return report
        requested = max(1, math.ceil(math.exp(log_requested)))
        report.requested_samples = requested
        # Protect against an upward integer rounding at the log-space boundary.
        if requested > max_samples:
            report.status = "sample_budget_exceeded"
            report.error = (
                "Required independent sample count exceeds per-order budget; no draws used")
            return report
        epsilon = math.exp(.5*(log_v - math.log(requested) - math.log(delta)))
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise FloatingPointError("Confidence radius exceeds float64 range")
        while report.samples < requested:
            budget.check()
            count = min(batch_size, requested - report.samples)
            g = sampler.sample(count)
            if len(g) != count:
                raise FloatingPointError("Sampler returned an unexpected batch size")
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                estimate = batch_moments(u, k, g, n=sampler.basis.n)
                # Only C is accumulated. q is recovered by Euler's identity.
                value = estimate.C * count - correction
                updated = total + value
                next_correction = (updated - total) - value
            if not np.all(np.isfinite(updated)):
                raise FloatingPointError("Nonfinite reconstruction accumulator")
            total, correction = updated, next_correction
            report.samples += count
        budget.check()
        diagnostics()
        if report.q_estimate is None or report.residual_estimate is None:
            raise FloatingPointError("Nonfinite reconstruction diagnostics")
        q_radius = norm * epsilon
        lower = max(deterministic_lower, report.q_estimate - q_radius)
        upper = min(deterministic_upper, report.q_estimate + q_radius)
        residual_bound = report.residual_estimate + epsilon
        if not all(math.isfinite(x) for x in (q_radius, lower, upper, residual_bound)):
            raise FloatingPointError("Confidence bounds exceed float64 range")
        report.statistical_radius = epsilon
        if lower > upper:
            report.status = "inconsistent_bounds"
            report.error = (
                "Euler confidence interval and deterministic moment bounds have no intersection")
            return report
        report.q_lower, report.q_upper = lower, upper
        report.residual_bound = residual_bound
        report.status = "certified"
        return report
    except (ResourceLimitError, FloatingPointError, OverflowError) as exc:
        report.status = ("resource_limit" if isinstance(exc, ResourceLimitError)
                         else "numerical_error")
        report.error = str(exc)
        diagnostics()
        return report
