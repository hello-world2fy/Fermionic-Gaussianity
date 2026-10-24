"""Adaptive-temperature SMC with exact one-plane Gaussian heat-bath moves.

The conditional Haar measure on exp(theta gamma_a gamma_b) g is uniform in
theta. Its overlap is a*cos(theta)+b*sin(theta). Diagonalizing this two by
two real quadratic form supplies a rejection sampler for the exact tempered
conditional density at ANY real beta >= 0. Each move preserves pi_beta;
no guessed Gaussian family or symmetry projection is used in sampling.

Adaptive SMC normalization and uncertainty are NUMERICAL estimates, not
interval certificates. Particle ESS is not the number of independent draws.
"""
from dataclasses import dataclass
from functools import lru_cache
import os
import time

import numpy as np
from numba import njit, prange
from scipy.special import logsumexp

from . import paths  # noqa: F401
from fk.gaussian import sample_gaussian, _bivector_action


@lru_cache(None)
def actions(n):
    pairs = [(a, b) for a in range(2*n) for b in range(a+1, 2*n)]
    values = [_bivector_action(n, *pair) for pair in pairs]
    return np.array([v[0] for v in values]), np.array([v[1] for v in values])


@njit(cache=True, parallel=True)
def _heatbath(kets, witness, beta, permutations, phases, planes, seeds):
    """Per-particle PRNG seeds make results independent of thread scheduling."""
    trials = np.zeros(len(kets), np.int64)
    for i in prange(len(kets)):
        np.random.seed(seeds[i])
        ket = kets[i].copy()
        work = np.empty_like(ket)
        a = 0j
        for j in range(len(ket)):
            a += witness[j].conjugate()*ket[j]
        for p in planes:
            b = 0j
            for j in range(len(ket)):
                work[j] = phases[p, j]*ket[permutations[p, j]]
                b += witness[j].conjugate()*work[j]
            aa, bb = abs(a)**2, abs(b)**2
            cross = (a.conjugate()*b).real
            center = .5*np.arctan2(2*cross, aa-bb)
            spread = np.hypot(aa-bb, 2*cross)
            largest = .5*(aa+bb+spread)
            smallest = max(0., .5*(aa+bb-spread))
            while True:
                angle = np.random.uniform(-np.pi, np.pi)
                trials[i] += 1
                ratio = (smallest+(largest-smallest)*np.cos(angle)**2)/largest if largest > 0 else 1.
                if beta == 0 or np.log(max(np.random.random(), 1e-300)) <= beta*np.log(max(ratio, 1e-300)):
                    break
            angle += center
            c, s = np.cos(angle), np.sin(angle)
            for j in range(len(ket)):
                ket[j] = c*ket[j]+s*work[j]
            a = c*a+s*b
        kets[i] = ket
    return trials


def heatbath(kets, witness, beta, rng, *, sweeps=1, n=8):
    permutations, phases = actions(n)
    planes = np.concatenate([rng.permutation(len(permutations)) for _ in range(sweeps)])
    seeds = rng.integers(0, 2**32-1, len(kets), dtype=np.uint32)
    trials = _heatbath(kets, witness, beta, permutations, phases, planes, seeds)
    return {"planes_per_particle": len(planes), "angle_trials_mean": float(trials.mean()),
            "exact_conditional_updates": True}


def log_overlap_squared(kets, witness):
    a = np.einsum("ni,i->n", kets.conj(), witness, optimize=False)
    return 2*np.log(np.maximum(abs(a), 1e-300))


def incremental_ess(log_r, step):
    v = step*log_r
    return float(np.exp(2*logsumexp(v)-logsumexp(2*v)))


def next_temperature(log_r, beta, endpoint, ess_fraction):
    if incremental_ess(log_r, endpoint-beta) >= ess_fraction*len(log_r):
        return float(endpoint)
    lo, hi = beta, endpoint
    for _ in range(45):
        mid = .5*(lo+hi)
        if incremental_ess(log_r, mid-beta) < ess_fraction*len(log_r):
            hi = mid
        else:
            lo = mid
    if lo-beta < 1e-12:
        raise ArithmeticError("temperature ladder stalled")
    return float(lo)


def systematic_resample(weights, rng):
    positions = (rng.random()+np.arange(len(weights)))/len(weights)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    return np.searchsorted(cdf, positions)


@dataclass
class TemperedResult:
    kets: np.ndarray
    log_z: float
    beta: float
    log_r: np.ndarray
    diagnostics: dict


def sample_tempered(witness, endpoint, count, rng, *, n=8, ess_fraction=.8,
                    sweeps=2, callback=None, checkpoints=(), state_callback=None,
                    log_z_anchors=None):
    backend = os.environ.get('FK_SAMPLER_BACKEND', 'cpu')
    if backend == 'cuda':
        from .gpu_tempering import sample_tempered_gpu
        return sample_tempered_gpu(witness, endpoint, count, rng, n=n,
                                   ess_fraction=ess_fraction, sweeps=sweeps,
                                   callback=callback, checkpoints=checkpoints,
                                   state_callback=state_callback, log_z_anchors=log_z_anchors,
                                   device=int(os.environ.get('FK_CUDA_DEVICE', '0')))
    if backend != 'cpu':
        raise ValueError('FK_SAMPLER_BACKEND must be cpu or cuda')
    if not 0 < ess_fraction < 1 or count < 2 or endpoint < 0 or sweeps < 1:
        raise ValueError("invalid tempering settings")
    u = np.asarray(witness, complex)
    if u.shape != (2**(n-1),) or not np.isfinite(u).all() or np.linalg.norm(u) == 0:
        raise ValueError("expected a finite nonzero even-sector witness")
    started = time.monotonic()
    kets = sample_gaussian(n, count, rng).kets.copy()
    log_r = log_overlap_squared(kets, u)
    ancestors = np.arange(count)
    beta, log_z = 0., 0.
    ladder = []
    anchors = {} if log_z_anchors is None else dict(log_z_anchors)
    stops = sorted(set([float(x) for x in (*checkpoints, *anchors) if 0 < x <= endpoint]+[float(endpoint)]))
    stop_index = 0
    while beta < endpoint-1e-12:
        stop = stops[stop_index]
        new_beta = next_temperature(log_r, beta, stop, ess_fraction)
        log_increment = (new_beta-beta)*log_r
        norm = float(logsumexp(log_increment))
        log_z += norm-np.log(count)
        weights = np.exp(log_increment-norm)
        ess = float(1/np.sum(weights**2))
        chosen = systematic_resample(weights, rng)
        kets = np.ascontiguousarray(kets[chosen])
        ancestors = ancestors[chosen]
        move = heatbath(kets, u, new_beta, rng, sweeps=sweeps, n=n)
        log_r = log_overlap_squared(kets, u)
        beta = new_beta
        row = {"beta": beta, "log_z": log_z, "incremental_ess": ess,
               "unique_initial_ancestors": len(np.unique(ancestors)),
               "log_r_mean": float(log_r.mean()), "log_r_std": float(log_r.std()),
               "elapsed_seconds": time.monotonic()-started, **move}
        if beta in anchors:
            row["log_z_before_analytic_anchor"] = log_z
            row["analytic_anchor_log_z"] = float(anchors[beta])
            log_z = float(anchors[beta])
            row["log_z"] = log_z
        ladder.append(row)
        if callback:
            callback(row)
        if beta >= stop-1e-12:
            if state_callback:
                state_callback(kets, beta, log_z, log_r, row)
            stop_index += 1
    norm_error = float(np.max(abs(np.linalg.norm(kets, axis=1)-1)))
    if norm_error > 1e-9:
        raise ArithmeticError("Gaussian unitary chain lost normalization")
    diagnostics = {"method": "adaptive_tempered_smc_exact_plane_heatbath",
                   "count": count, "sweeps_per_temperature": sweeps,
                   "ess_fraction": ess_fraction, "ladder": ladder,
                   "max_norm_error": norm_error, "elapsed_seconds": time.monotonic()-started,
                   "analytic_log_z_anchors": anchors,
                   "iid_particles": False, "strictly_certified": False}
    return TemperedResult(kets, log_z, beta, log_r, diagnostics)
