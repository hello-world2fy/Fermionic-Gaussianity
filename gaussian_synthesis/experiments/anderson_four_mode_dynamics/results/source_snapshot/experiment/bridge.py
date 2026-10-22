"""Actual calls to the original convex synthesis and its continuous measure.

The exact reference trajectory is deliberately absent from this module.
Two-node compression preserves C_2(u), not the full measure or its total mass.
"""
import bootstrap
import time
import numpy as np
from gaussian_synthesis import SynthesisConfig, synthesize
from gaussian_synthesis.measure import GaussianCoefficientMeasure
from gaussian_synthesis.sampling import HaarFourModeSampler
from gaussian_synthesis.solver import projected_sgd, ResourceBudget
from gaussian_synthesis.gaussian import spin_to_fock
from exact_solver import solve_exact, exact_k2, exact_hessian
from model import BASIS, GaussianSum, PHYSICAL, schmidt_decompose

SPIN_TO_FOCK = spin_to_fock(np.eye(8)).T


def solve_measure(target, backend, seed, initial=None):
    """No reimplementation of either repository solver here."""
    start = time.perf_counter()
    records = {'backend': backend, 'seed': int(seed), 'orders': [2]}
    if backend in ('api', 'api_polished'):
        config = SynthesisConfig(orders=(2,), seed=int(seed), iterations=800,
            training_batch=256, diagnostic_samples=512, validation_batch=4096,
            k1_mass_samples=1024, tolerance=.1, max_validation_samples=1_000_000)
        result = synthesize(target, BASIS, config)
        if not result.stages:
            raise RuntimeError(f'Original synthesize returned no stage: {result.status}')
        stage = result.stages[0]
        measure = stage.measure
        records.update(api_status=result.status, api_summary=result.summary(),
                       source_function='gaussian_synthesis.api.synthesize',
                       iterations=stage.solution.iterations,
                       training_samples=stage.solution.iterations*config.training_batch,
                       diagnostic_samples=len(stage.solution.history)*config.diagnostic_samples,
                       validation_samples=stage.validation.samples if stage.validation else 0)
    elif backend in ('sgd', 'sgd_polished'):
        children = np.random.SeedSequence(seed).spawn(2)
        samplers = [HaarFourModeSampler(np.random.default_rng(child), BASIS) for child in children]
        result = projected_sgd(target, 2, samplers[0], samplers[1], ResourceBudget(),
            initial=initial, iterations=256, batch_size=128, diagnostic_samples=256)
        if result.status != 'completed':
            raise RuntimeError(result.status)
        measure = GaussianCoefficientMeasure(BASIS, 2, result.u)
        records.update(source_function='gaussian_synthesis.solver.projected_sgd',
                       iterations=result.iterations, solver_status=result.status,
                       training_samples=256*128, diagnostic_samples=512)
    elif backend == 'newton':
        result = solve_exact(target)
        if result.status != 'converged':
            raise RuntimeError(result.status)
        measure = GaussianCoefficientMeasure(BASIS, 2, result.u)
        records.update(source_function='four_mode_landscape.exact_solver.solve_exact',
                       iterations=result.iterations, solver_status=result.status,
                       training_samples=0, diagnostic_samples=0)
    else:
        raise ValueError(backend)
    if backend.endswith('_polished'):
        u = measure.u.copy()
        records['residual_before_polish'] = float(np.linalg.norm(exact_k2(u)[1]-target))
        for iteration in range(20):
            q, moment = exact_k2(u)
            gradient = moment-target
            if np.linalg.norm(gradient) < 1e-11:
                break
            real_gradient = np.r_[gradient.real, gradient.imag]
            direction = np.linalg.solve(exact_hessian(u), -real_gradient)
            direction = direction[:8]+1j*direction[8:]
            value = q/4-np.vdot(u, target).real
            derivative = np.vdot(gradient, direction).real
            step = 1.
            for _ in range(40):
                candidate = u+step*direction
                candidate_value = exact_k2(candidate)[0]/4-np.vdot(candidate, target).real
                if candidate_value <= value+1e-4*step*derivative+1e-14:
                    u = candidate
                    break
                step /= 2
            else:
                raise RuntimeError('Witness polish line search failed')
        measure = GaussianCoefficientMeasure(BASIS, 2, u)
        records['polish_newton_steps'] = iteration
        records['polish_source'] = 'original exact_k2 and exact_hessian; initialized at original SGD witness'
    records['solve_seconds'] = time.perf_counter()-start
    _, reconstructed = exact_k2(measure.u)
    records['continuous_residual'] = float(np.linalg.norm(target-reconstructed))
    return measure, records


def sampled_discretization(measure, count, seed, *, importance=True):
    """Exact rejection sampling from density proportional to |<g|u>|^3.

    Original Haar sampler + original measure mass estimator. The mass estimate
    is independent of accepted atoms. Final coefficients may subsequently be
    refitted. Every proposed sample and all mass-estimation work are charged.
    """
    start = time.perf_counter()
    roots = np.random.SeedSequence(seed).spawn(3)
    sampler = HaarFourModeSampler(np.random.default_rng(roots[0]), BASIS)
    mass_sampler = HaarFourModeSampler(np.random.default_rng(roots[1]), BASIS)
    random = np.random.default_rng(roots[2])
    if importance:
        mass_info = measure.estimate_mass(mass_sampler, 1024, batch_size=1024)
        mass = mass_info['estimate']
        norm = np.linalg.norm(measure.u)
        accepted = []
        proposals = 0
        total = 0
        while total < count:
            batch = sampler.sample(512)
            overlaps = batch.conj() @ measure.u
            probabilities = (np.abs(overlaps)/norm)**(2*measure.k-1)
            chosen = batch[random.random(len(batch)) < probabilities]
            accepted.append(chosen)
            proposals += len(batch)
            total += len(chosen)
            if proposals > 131072:
                raise RuntimeError('Small rejection-sampling budget exhausted')
        samples = np.concatenate(accepted)[:count]
        overlaps = samples.conj() @ measure.u
        phase = np.divide(overlaps, np.abs(overlaps), out=np.ones_like(overlaps), where=np.abs(overlaps)>0)
        atoms = (samples*phase[:, None]).T
        coefficients = np.full(count, mass/count, complex)
        info = {'proposal_samples': proposals, 'mass_samples': 1024,
                'mass_estimate': mass, 'mass_standard_error': mass_info['standard_error']}
    else:
        atoms = sampler.sample(count).T
        coefficients = np.zeros(count, complex)
        info = {'proposal_samples': count, 'mass_samples': 0, 'mass_estimate': None}
    info.update(sampling_seconds=time.perf_counter()-start, accepted_samples=count,
                density='original GaussianCoefficientMeasure, k=2' if importance else 'uniform Haar')
    return GaussianSum(atoms, coefficients), info


def two_node_discretization(measure):
    """Deterministic first-vector-moment compression of the k=2 measure.

    Derived from the witness, not the target Schmidt decomposition. If SGD
    has not solved C_2(u)=target, the two-node state retains that solver error.
    """
    start = time.perf_counter()
    if measure.k != 2:
        raise ValueError('This exact moment compression is specific to k=2')
    z = SPIN_TO_FOCK.conj().T @ measure.u
    trace = z @ z
    phase = .5*np.angle(trace) if abs(trace)>1e-14 else 0.
    rotated = np.exp(-1j*phase)*z
    r = rotated.real
    s = rotated.imag
    r /= np.linalg.norm(r)
    s -= r*np.dot(r, s)
    if np.linalg.norm(s) < 1e-12:
        s = np.eye(8)[int(np.argmin(np.abs(r)))].copy()
        s -= r*np.dot(r, s)
    s /= np.linalg.norm(s)
    atoms = SPIN_TO_FOCK @ np.column_stack([(r+1j*s)/np.sqrt(2), (r-1j*s)/np.sqrt(2)])
    _, moment = exact_k2(measure.u)
    coefficients = atoms.conj().T @ moment
    keep = np.abs(coefficients)>1e-13
    state = GaussianSum(atoms[:, keep], coefficients[keep])
    return state, {'sampling_seconds': 0., 'moment_compression_seconds': time.perf_counter()-start,
                   'proposal_samples': 0, 'mass_samples': 0, 'accepted_samples': 0,
                   'moment_preservation_residual': float(np.linalg.norm(state.vector()-moment)),
                   'preserves': 'first vector moment C_2(u); NOT total mass or full measure'}


def symmetry_dictionary(state):
    """Project to N_up=N_down=1, then split each projected ket into Slaters."""
    atoms = []
    coefficients = []
    for atom, coefficient in zip(state.atoms.T, state.coefficients):
        pieces = schmidt_decompose(atom)
        atoms.append(pieces.atoms)
        coefficients.extend(coefficient*pieces.coefficients)
    return GaussianSum(np.concatenate(atoms, axis=1), np.array(coefficients))


def refit(dictionary, target, cap=8, tolerance=1e-10):
    """Coherent OMP and complex least squares; selected atoms are unchanged."""
    atoms = dictionary.atoms
    residual = target.copy()
    selected = []
    coefficients = np.empty(0, complex)
    for _ in range(min(cap, atoms.shape[1])):
        scores = np.abs(atoms.conj().T @ residual)
        scores[selected] = -1.
        index = int(np.argmax(scores))
        selected.append(index)
        coefficients = np.linalg.lstsq(atoms[:, selected], target, rcond=1e-12)[0]
        residual = target-atoms[:, selected] @ coefficients
        if np.linalg.norm(residual) <= tolerance:
            break
    state = GaussianSum(atoms[:, selected].copy(), coefficients)
    return state, {'refit_residual': float(np.linalg.norm(residual)),
                   'selected_atoms': len(selected), 'dictionary_atoms': atoms.shape[1]}
