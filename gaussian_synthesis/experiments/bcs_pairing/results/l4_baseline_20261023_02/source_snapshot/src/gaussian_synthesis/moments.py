"""Log-scaled moments with streaming, compensated accumulation."""

from dataclasses import dataclass
import math
import numpy as np
from .theory import check_order, log_dimension


@dataclass
class MomentEstimate:
    q: float
    C: np.ndarray
    mass: float
    second_mass: float
    samples: int

    @property
    def mass_standard_error(self):
        if self.samples < 2:
            return None
        return math.sqrt(max(0., self.second_mass-self.mass**2)/(self.samples-1))


def _rescale(value, log_scale):
    """Rescale only after reduction, retaining complex signs/phases."""
    value = np.asarray(value)
    size = np.abs(value)
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = log_scale + np.log(size)
    if np.any(exponent > np.log(np.finfo(float).max)):
        raise FloatingPointError("Moment exceeds float64 range; reduce k or vector scale")
    phase = np.divide(value, size, out=np.zeros_like(value), where=size != 0)
    with np.errstate(under="ignore"):
        result = phase*np.exp(exponent)
    return result.item() if result.ndim == 0 else result


def scalar_moment(u, power, samples, *, log_prefactor=0.):
    """Estimate prefactor * E |<g|u>|^power on the supplied batch."""
    overlaps = np.einsum("bi,i->b", samples.conj(), u)
    magnitude = np.abs(overlaps)
    with np.errstate(divide="ignore"):
        logs = log_prefactor + power*np.log(magnitude)
    scale = float(np.max(logs))
    if scale == -math.inf:
        return 0.
    return float(_rescale(np.mean(np.exp(logs-scale)), scale))


def batch_moments(u, k, samples, *, n=4):
    check_order(k)
    u = np.asarray(u, dtype=np.complex128)
    samples = np.asarray(samples, dtype=np.complex128)
    if (samples.ndim != 2 or not 1 <= len(samples) <= 8192
            or samples.shape[1:] != u.shape or u.shape != (2**(n-1),)):
        raise ValueError("Invalid or oversized Gaussian sample batch")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(u)):
        raise ValueError("Moments require finite vectors")
    a = np.einsum("bi,i->b", samples.conj(), u)
    magnitude = np.abs(a)
    phase = np.divide(a, magnitude, out=np.zeros_like(a), where=magnitude != 0)
    with np.errstate(divide="ignore"):
        log_mass = log_dimension(n, k)+(2*k-1)*np.log(magnitude)
    scale = float(np.max(log_mass))
    if scale == -math.inf:
        return MomentEstimate(0., np.zeros_like(u), 0., 0., len(samples))
    weights = np.exp(log_mass-scale)
    C = _rescale(np.einsum("b,bi->i", weights*phase, samples)/len(samples), scale)
    q = float(_rescale(np.mean(weights*magnitude), scale))
    mass = float(_rescale(np.mean(weights), scale))
    second = float(_rescale(np.mean(weights**2), 2*scale))
    return MomentEstimate(q, C, mass, second, len(samples))


class MomentAccumulator:
    """O(D) storage regardless of the total number of draws."""

    def __init__(self, dimension):
        self.count = 0
        self.total = np.zeros(dimension+3, dtype=np.complex128)
        self.correction = np.zeros_like(self.total)

    def add(self, estimate):
        packed = np.concatenate(([estimate.q, estimate.mass, estimate.second_mass], estimate.C))
        value = packed*estimate.samples-self.correction
        updated = self.total+value
        self.correction = (updated-self.total)-value
        self.total = updated
        self.count += estimate.samples

    def result(self):
        if not self.count:
            raise ValueError("No samples accumulated")
        mean = self.total/self.count
        return MomentEstimate(float(mean[0].real), mean[3:].copy(),
                              float(mean[1].real), float(mean[2].real), self.count)


def estimate_moments(u, k, sampler, count, *, batch_size=8192, check=None):
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Sample count must be positive")
    if not 1 <= batch_size <= 8192:
        raise ValueError("batch_size must be between 1 and 8192")
    accumulator = MomentAccumulator(len(u))
    while accumulator.count < count:
        if check:
            check()
        g = sampler.sample(min(batch_size, count-accumulator.count))
        accumulator.add(batch_moments(u, k, g, n=sampler.basis.n))
    if check:
        check()
    return accumulator.result()
