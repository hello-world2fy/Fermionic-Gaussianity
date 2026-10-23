"""Witness-guided Gaussian discovery and finite-orbit cost comparisons.

No structured ansatz enters ``optimize_gaussian_overlap``. It acts on Haar
starts using all 120 even Majorana bivectors, hence explores the full even
Gaussian manifold. Cyclic plane coordinate ascent maximizes each plane's
overlap exactly; it is a bounded local search, NOT a global overlap oracle.

Every method gets the same 128 Fock atoms plus a fixed number of new atoms.
Random and guided methods share their initial Haar atoms. Candidate counts
are equal, but the methods do not have equal wall-clock cost. Physics-arc
atoms are an explicitly labelled prior, constructed only in their own branch.

Saved array prefixes are ``{method}_t{t:g}`` (decimal points become ``p``):
atoms are normalized Gaussian ROW kets, phases multiply the unaligned raw
new atoms, covariance/invariants describe all atoms, projected_atoms are
unnormalized COLUMN coordinates, and coefficients multiply those columns.
The represented objects are Spin(7) orbit averages, not finite Gaussian sums.
All costs include the full 128-dimensional Fock residual repair. Target-state
error corrections and separation claims remain floating-point evidence.
"""
from functools import lru_cache
import time

import numpy as np
from scipy.linalg import eigh

from fk.baselines import solve_projected_atomic_l1, target_error_estimate
from fk.certificates import separation_certificate
from fk.gaussian import _bivector_action, sample_gaussian
from fk.observables import atom_invariants


@lru_cache(maxsize=None)
def bivector_actions(n=8):
    pairs = tuple((a, b) for a in range(2*n) for b in range(a+1, 2*n))
    actions = tuple(_bivector_action(n, a, b) for a, b in pairs)
    return pairs, np.asarray([x[0] for x in actions]), np.asarray([x[1] for x in actions])


def rotate_plane(kets, a, b, angles, n=8):
    """Apply exp(theta gamma_a gamma_b); theta is the spin half-angle."""
    states = np.asarray(kets, dtype=complex)
    permutation, phase = _bivector_action(n, a, b)
    theta = np.asarray(angles)
    return np.cos(theta)[..., None]*states + np.sin(theta)[..., None]*states[..., permutation]*phase


def overlap_gradient(witness, kets, n=8):
    """Real derivatives of |<u|exp(theta G_p)|g>|² at theta=0."""
    u = np.asarray(witness, dtype=complex)
    states = np.atleast_2d(np.asarray(kets, dtype=complex))
    _, permutations, phases = bivector_actions(n)
    transformed = states[:, permutations]*phases[None, :, :]
    a = np.einsum('j,bj->b', u.conj(), states, optimize=False)
    b = np.einsum('j,bpj->bp', u.conj(), transformed, optimize=False)
    return 2*np.real(a.conj()[:, None]*b)


def covariance_from_kets(kets, n=8):
    """Compute covariances directly from signed bivector matrix elements."""
    states = np.atleast_2d(np.asarray(kets, dtype=complex))
    pairs, permutations, phases = bivector_actions(n)
    transformed = states[:, permutations]*phases[None, :, :]
    expectations = np.einsum('bj,bpj->bp', states.conj(), transformed, optimize=False)
    covariances = np.zeros((len(states), 2*n, 2*n))
    for index, (a, b) in enumerate(pairs):
        covariances[:, a, b] = (1j*expectations[:, index]).real
        covariances[:, b, a] = -covariances[:, a, b]
    return covariances


def optimize_gaussian_overlap(witness, starts, *, max_sweeps=30,
                              gradient_tolerance=1e-8, budget=None, n=8):
    """Batched cyclic coordinate ascent with exact one-plane maximizers.

    Writing a=<u|g>, b=<u|Gg>, the plane objective is
    |a|² cos²(theta)+|b|² sin²(theta)+2 Re(a* b) cos(theta)sin(theta).
    Its maximizing angle is 0.5 atan2(2 Re(a* b), |a|²-|b|²).
    We normalize u internally so convergence tolerances do not depend on its
    arbitrary magnitude. Phases of u and of each start cannot affect the path
    in ray space. The full rotation path is saved for independent replay.
    """
    raw_witness = np.asarray(witness, dtype=complex)
    states = np.array(starts, dtype=complex, copy=True)
    if states.ndim != 2 or states.shape[1] != 2**(n-1):
        raise ValueError('starts must be even Gaussian row kets')
    if raw_witness.shape != (states.shape[1],) or not np.all(np.isfinite(raw_witness)):
        raise ValueError('witness must be a finite full even-sector ket')
    if not np.allclose(np.linalg.norm(states, axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError('starts must be normalized')
    if max_sweeps < 1 or gradient_tolerance <= 0:
        raise ValueError('positive iteration budget and tolerance required')
    witness_norm = float(np.linalg.norm(raw_witness))
    if witness_norm == 0:
        raise ValueError('zero witness gives no discovery direction')
    u = raw_witness/witness_norm
    pairs, permutations, phases = bivector_actions(n)
    objective = lambda: abs(np.einsum('j,bj->b', u.conj(), states, optimize=False))**2
    trace = [objective()]
    rotations = []
    converged = np.zeros(len(states), dtype=bool)
    started = time.monotonic()
    for sweep in range(max_sweeps):
        if budget is not None:
            budget.check()
        angles = np.zeros((len(states), len(pairs)))
        for p, (permutation, phase) in enumerate(zip(permutations, phases)):
            transformed = states[:, permutation]*phase
            a = np.einsum('j,bj->b', u.conj(), states, optimize=False)
            b = np.einsum('j,bj->b', u.conj(), transformed, optimize=False)
            theta = .5*np.arctan2(2*np.real(a.conj()*b), abs(a)**2-abs(b)**2)
            theta[converged] = 0
            states = np.cos(theta)[:, None]*states + np.sin(theta)[:, None]*transformed
            angles[:, p] = theta
        trace.append(objective())
        rotations.append(angles)
        gradient = overlap_gradient(u, states, n)
        converged = np.max(abs(gradient), axis=1) <= gradient_tolerance
        if np.all(converged):
            break
    norm_error = float(np.max(abs(np.linalg.norm(states, axis=1)-1)))
    if norm_error > 1e-10:
        raise RuntimeError('Gaussian rotation chain lost normalization')
    # Only a final scalar roundoff correction; never project onto a guessed
    # physical family or an invariant subspace during discovery.
    states /= np.linalg.norm(states, axis=1)[:, None]
    gradient = overlap_gradient(u, states, n)
    return states, {
        'objective_trace_normalized_witness': np.asarray(trace),
        'rotation_angles': np.asarray(rotations),
        'rotation_pairs': np.asarray(pairs, dtype=int),
        'gradient_final_normalized_witness': gradient,
        'converged': np.max(abs(gradient), axis=1) <= gradient_tolerance,
        'witness_norm': witness_norm,
        'elapsed_seconds': time.monotonic()-started,
        'sweeps': len(rotations),
        'plane_count_per_sweep': len(pairs),
        'max_norm_drift_before_correction': norm_error,
        'globally_optimized': False,
    }


def phase_align(witness, atoms):
    overlaps = np.einsum('j,bj->b', np.asarray(witness).conj(), atoms, optimize=False)
    phases = np.ones(len(atoms), dtype=complex)
    nonzero = abs(overlaps) > 1e-15
    phases[nonzero] = overlaps[nonzero].conj()/abs(overlaps[nonzero])
    return np.asarray(atoms)*phases[:, None], phases


def physics_arc_atoms(model, count, budget):
    """Explicit physical prior, intentionally separate from guided search."""
    masks = np.asarray(model.even_masks, dtype=int)
    qa = model.q_ops[0, 0][np.ix_(masks, masks)]
    qb = model.q_ops[1, 0][np.ix_(masks, masks)]
    hop = model.h_hop[np.ix_(masks, masks)]
    angles = np.linspace(0, np.pi, count)
    atoms = []
    for theta in angles:
        budget.check()
        _, vectors = eigh(np.cos(theta)*(-qa+qb)+np.sin(theta)*hop,
                          subset_by_index=(0, 0), check_finite=False)
        atoms.append(vectors[:, 0])
    return np.asarray(atoms), angles


def run_discovery(model, witnesses, config, seed, budget, checkpoint=None):
    """Compare paired random, canonical-guided and physics-prior dictionaries.

    ``witnesses`` maps numeric t values to full even128 kets. Optional settings
    in config['discovery']: atom_count, max_sweeps, gradient_tolerance,
    eta_thresholds. Caller owns the stage/global resource budget and checkpoint.
    """
    settings = config.get('discovery', {})
    count = int(settings.get('atom_count', 32))
    if count < 2:
        raise ValueError('At least two added atoms required')
    eta_thresholds = settings.get('eta_thresholds', [0., .25, .5, .75])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 910, 3]))
    started = time.monotonic()
    report = {
        'status': 'running', 'seed': int(seed), 'rows': [],
        'rng_seed_sequence': [int(seed), 910, 3],
        'common_fock_atoms': 128, 'added_atoms_per_method': count,
        'same_candidate_count': True, 'same_wallclock_budget_per_method': False,
        'guide': 'Frozen canonical k2 witness only; all120 Majorana planes; no arc initialization',
        'search': 'Cyclic exact plane maximization, bounded local optimization, no global oracle',
        'scope': 'Finite Spin(7) orbit dictionary; projected atoms are not renormalized',
        'global_optimality_certified': False, 'local_search_is_dual_certificate': False,
        'strict_interval_certified': False, 'statistically_certified': False,
    }
    arrays = {'fock_atoms': np.eye(128, dtype=complex)}
    fock_covariance = covariance_from_kets(arrays['fock_atoms'])
    prior_started = time.monotonic()
    prior_atoms, prior_angles = physics_arc_atoms(model, count, budget)
    prior_seconds = time.monotonic()-prior_started
    arrays['physics_arc_angles'] = prior_angles
    arrays['physics_arc_raw_atoms'] = prior_atoms
    report['physics_prior_construction_seconds_once'] = prior_seconds
    for t in config['t_values']:
        budget.check()
        t = float(t)
        tkey = f't{t:g}'.replace('.', 'p')
        witness = np.asarray(witnesses[t], dtype=complex)
        target = np.asarray(model.target(t))
        target_diagnostics = target_error_estimate(model, t, target)
        state_error = target_diagnostics['target_state_error_estimate']
        schmidt = float(np.linalg.svd(model.embed(target).reshape(16, 16), compute_uv=False).sum())
        sample_started = time.monotonic()
        sample = sample_gaussian(8, count, rng)
        sample_seconds = time.monotonic()-sample_started
        arrays[tkey+'_haar_starts'] = sample.kets
        arrays[tkey+'_witness'] = witness
        arrays[tkey+'_target'] = target
        guided, trace = optimize_gaussian_overlap(
            witness, sample.kets, max_sweeps=int(settings.get('max_sweeps', 30)),
            gradient_tolerance=float(settings.get('gradient_tolerance', 1e-8)), budget=budget)
        for field, value in trace.items():
            if isinstance(value, np.ndarray):
                arrays[tkey+'_guided_'+field] = value
        for name, raw, generation_seconds in (
                ('random', sample.kets, sample_seconds),
                ('guided', guided, sample_seconds+trace['elapsed_seconds']),
                ('physics_arc', prior_atoms, prior_seconds)):
            budget.check()
            aligned, phases = phase_align(witness, raw)
            atoms = np.concatenate((arrays['fock_atoms'], aligned))
            covariances = np.concatenate((fock_covariance, covariance_from_kets(aligned)))
            norm_error = float(np.max(abs(np.linalg.norm(atoms, axis=1)-1)))
            purity_error = float(np.max(abs(covariances@covariances+np.eye(16))))
            if max(norm_error, purity_error) > 1e-9:
                raise RuntimeError('Candidate normalization or pure Gaussianity gate failed')
            inv = atom_invariants(covariances, model.j_matrices)
            # Post hoc comparison with the prior's invariant curve only.
            # This diagnostic never enters optimization or atom selection;
            # it is not Hilbert distance to an orbit of physical arc states.
            features = np.column_stack([inv['mass_a'], inv['mass_b'],
                                        inv['mass_dot'], inv['eta']])
            intercept = np.array([1., 1., -1., 0.])
            direction = np.array([-1., -1., 1., 1.])
            arc_parameter = np.clip((features-intercept)@direction/4, 0, 1)
            inv['saturation_arc_invariant_distance'] = np.linalg.norm(
                features-intercept-arc_parameter[:, None]*direction, axis=1)
            solve_started = time.monotonic()
            solved = solve_projected_atomic_l1(model.S, atoms, target)
            solve_seconds = time.monotonic()-solve_started
            certificates = [separation_certificate(solved['coefficient_l1'], solved['residual'],
                                                   schmidt, eta0=eta,
                                                   state_error=state_error)
                            for eta in eta_thresholds]
            key = name+'_'+tkey
            arrays[key+'_atoms'] = atoms
            arrays[key+'_new_raw_atoms'] = raw
            arrays[key+'_phases'] = phases
            arrays[key+'_covariances'] = covariances
            for field in ('coefficients', 'projected_atoms', 'reconstruction', 'residual'):
                arrays[key+'_'+field] = solved[field]
            for field, value in inv.items():
                arrays[key+'_'+field] = value
            new_mass = abs(solved['coefficients'][128:])
            total = float(np.sum(abs(solved['coefficients'])))
            overlap_squared = abs(np.einsum('j,bj->b', witness.conj(), aligned, optimize=False))**2
            row = {
                'key': key, 't': t, 'method': name, 'atom_count': len(atoms),
                'coefficient_l1': solved['coefficient_l1'],
                'cost_upper': certificates[0]['cost_upper'],
                'residual_l2': float(np.linalg.norm(solved['residual'])),
                'fock_residual_l1': float(np.sum(abs(solved['residual']))),
                'target_diagnostics': target_diagnostics,
                'certificates': certificates, 'schmidt_norm': schmidt,
                'solver_status': solved['solver_status'],
                'solver_optimality_scope': 'This finite orbit dictionary only',
                'active_atoms_1e_7': solved['active_atoms_1e_7'],
                'active_new_atoms_1e_7': int(np.sum(new_mass > 1e-7)),
                'new_atom_coefficient_mass_fraction': float(new_mass.sum()/total),
                'coefficient_mass_mean_eta': float(abs(solved['coefficients'])@inv['eta']/total),
                'coefficient_mass_mean_mass_a': float(abs(solved['coefficients'])@inv['mass_a']/total),
                'coefficient_mass_mean_arc_invariant_distance': float(
                    abs(solved['coefficients'])@inv['saturation_arc_invariant_distance']/total),
                'arc_distance_interpretation': 'Euclidean distance of (mass_a,mass_b,mass_dot,eta) to (1-x,1-x,-1+x,x); not Hilbert distance',
                'max_witness_overlap_squared': float(overlap_squared.max()),
                'mean_witness_overlap_squared': float(overlap_squared.mean()),
                'candidate_generation_seconds': float(generation_seconds),
                'candidate_timing_note': ('Prior generation is shared across t; time shown is once-only cost'
                                          if name == 'physics_arc' else 'Includes shared Haar initialization'),
                'coefficient_solve_seconds': float(solve_seconds),
                'atom_norm_error': norm_error, 'gaussian_purity_error': purity_error,
                'evidence_level': 'numerical_evidence',
                'global_optimality_certified': False, 'strict_interval_certified': False,
            }
            if name == 'guided':
                row.update({
                    'search_sweeps': trace['sweeps'],
                    'search_plane_rotations_per_start': trace['sweeps']*trace['plane_count_per_sweep'],
                    'search_converged_starts': int(np.sum(trace['converged'])),
                    'search_max_gradient': float(np.max(abs(trace['gradient_final_normalized_witness']))),
                    'search_is_global_or_dual_certificate': False,
                })
            report['rows'].append(row)
            if checkpoint:
                checkpoint(report, arrays)
    report['elapsed_seconds'] = float(time.monotonic()-started)
    report['status'] = 'complete'
    return report, arrays
