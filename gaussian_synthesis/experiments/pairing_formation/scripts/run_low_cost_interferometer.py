#!/usr/bin/env python3
"""Low-cost Gaussian decompositions and post-pairing interferometer readouts.

This round does not add k=2 labels or frozen interacting propagation.
Interaction prepares a seniority-zero state. The recorded tasks are:

1. Blank canonical mass, spectator invariance, reconstruction error, coefficient mass.
2. Alternative decompositions at the same budget, including an onsite-only ablation.
3. Whether a certified onsite-exclusion bound is nontrivial, and whether the
   readout bound B^2 alpha is nontrivial.
4. A family frozen on one training condition, then refit on new spectra,
   channel asymmetries and interferometer settings.

Eight H20 workers sample Haar atoms and run cross-level Gaussian search.
The remaining CPUs evaluate ansätze, branch-and-bound boxes and readouts.
Each process uses one numerical thread.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
import json
import math
import time
import hashlib
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.low_cost import (
    all_to_all, blank_coefficient_mass, canonical_mean_fidelity, edge_bunched_energies,
    embed_active_state, log_blank_coefficient_mass, log_frame_dimension, minimal_k_within_mass,
    outside_configuration_fraction, quartic_tilt_energies, spectator_extension, trivial_mass_lower_bound,
    certified_onsite_overlap)
from pairing_formation.interferometer import (
    count_errors, interferometer_readout, orbit_parity_element, transfer_expectation)
from pairing_formation.structure import analyze
from pairing_formation.fitting import fit_product_family
from pairing_formation.conditional import fit_conditional_products
from pairing_formation.jastrow import JastrowDynamics, fit_jastrow
from pairing_formation.costs import matched_magic_witness, repair_minimum_mass
from pairing_formation.replica2 import Replica2
from pairing_formation.pairing_channels import angular_model


TRAIN_LEVELS = 6
TRAIN_COUPLINGS = (2.7, 3.3, 4.8)
TRAIN_TIMES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
TV_GATE = 0.02
PARITY_GATE = 0.02
MASS_RATIO_TARGET = 1.2
GATE_PHASES = (0.0, math.pi / 8)
SURVEY_PHASES = (0.0, math.pi / 8, math.pi / 5)
SURVEY_ALPHAS = (math.pi / 4, math.pi / 6)
CHANNEL_STRENGTHS = (0.2, 0.37, 0.55, 0.73)
FAMILY_ORDER = (
    'single_direction', 'onsite_pbcs', 'conditional_orbitals', 'two_body_jastrow',
    'general_projected', 'adaptive_gaussian')
CPU_WORKERS = max(32, len(os.sched_getaffinity(0)) - 10)
GPU_COUNT = 8
HAAR_BATCH = 16384
HAAR_BATCHES = 8
ADAPTIVE_COUNT = 12
HOLDOUT_COUNT = 6


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def dump(path, payload):
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False))


def spec_seed(*parts):
    raw = json.dumps(parts, sort_keys=True, default=str)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16) % 10_000_000


def quartic_spec(levels, coupling):
    return {'kind': 'quartic', 'L': int(levels), 'G': float(coupling)}


def edge_spec(levels, coupling):
    return {'kind': 'edge', 'L': int(levels), 'G': float(coupling)}


def angular_spec(levels, coupling, relative):
    return {'kind': 'angular', 'L': int(levels), 'G': float(coupling), 'relative': float(relative)}


def make_model(spec):
    if spec['kind'] == 'quartic':
        return all_to_all(quartic_tilt_energies(spec['L']), spec['G'])
    if spec['kind'] == 'edge':
        return all_to_all(edge_bunched_energies(spec['L']), spec['G'])
    if spec['kind'] == 'angular':
        return angular_model(spec['L'], spec['G'], spec['relative'])[0]
    if spec['kind'] == 'spectator':
        active = make_model(spec['active'])
        return spectator_extension(active.energies, spec['active']['G'], gap=spec.get('gap', 4.0))
    raise ValueError(spec['kind'])


def default_port(levels):
    return ((0, 1), (levels - 2, levels - 1))


def crossed_port(levels):
    return ((0, 2), (levels - 3, levels - 1))


def band_ports(levels):
    half = levels // 2
    holes = [(a, b) for a in range(half) for b in range(a + 1, half)]
    particles = [(a, b) for a in range(half, levels) for b in range(a + 1, levels)]
    return [(h, p) for h in holes for p in particles]


def normalized(vector):
    vector = np.asarray(vector, complex)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-15:
        raise RuntimeError('encountered a zero vector')
    return vector / norm


def fidelity(left, right):
    return float(abs(np.vdot(left, right))**2)


def fast_approximations(model, state, seed):
    state = normalized(state)
    info, projected, _ = analyze(model, state)
    pbcs, pbcs_state = fit_product_family(model, state, radial=False, starts=8, seed=seed)
    conditional = fit_conditional_products(model, state, starts=8, seed=seed + 1)
    jastrow_report, _, jastrow_state = fit_jastrow(
        JastrowDynamics(model, 'two_body'), state, starts=6, seed=seed + 2)
    return {
        'single_direction': {
            'fidelity': float(info['single_direction_radial_fidelity']),
            'conditional_rank1_fidelity_upper': float(info['conditional_rank1_fidelity_upper']),
            'state': normalized(projected), 'global_optimum_certified': False},
        'onsite_pbcs': {
            'fidelity': float(pbcs['fidelity']), 'state': normalized(pbcs_state),
            'global_optimum_certified': False},
        'conditional_orbitals': {
            'fidelity': float(conditional[0]['fidelity']), 'state': normalized(conditional[3]),
            'global_optimum_certified': False},
        'two_body_jastrow': {
            'fidelity': float(jastrow_report['fidelity']), 'state': normalized(jastrow_state),
            'global_optimum_certified': False},
    }


def slim_readout(model, exact, approx, ports, phases, alphas):
    worst_tv, worst_parity = 0.0, 0.0
    rows, formula_errors = [], []
    for holes, particles in ports:
        for alpha in alphas:
            for phi in phases:
                reference = interferometer_readout(model, exact, holes, particles, phi, alpha)
                estimate = interferometer_readout(model, approx, holes, particles, phi, alpha)
                if reference['formula_count_error'] is not None:
                    formula_errors.append(reference['formula_count_error'])
                error = count_errors(reference, estimate)
                worst_tv = max(worst_tv, error['delta_count_tv'])
                worst_parity = max(worst_parity, error['absolute_delta_parity_error'])
                rows.append({
                    'holes': list(holes), 'particles': list(particles), 'phi': float(phi), 'alpha': float(alpha),
                    'exact_delta_parity': reference['delta_parity'], 'exact_raw_parity': reference['raw_parity'],
                    'exact_T': [reference['T_real'], reference['T_imag']],
                    'delta_count_tv': error['delta_count_tv'],
                    'absolute_delta_parity_error': error['absolute_delta_parity_error'],
                    'raw_count_tv': error['raw_count_tv'],
                    'formula_count_error': reference['formula_count_error']})
    balanced = [row for row in rows if abs(row['alpha'] - math.pi / 4) < 1e-12]
    gate_rows = balanced or rows
    gate_tv = max(row['delta_count_tv'] for row in gate_rows)
    gate_parity = max(row['absolute_delta_parity_error'] for row in gate_rows)
    return {
        'worst_delta_count_tv': worst_tv,
        'worst_abs_delta_parity_error': worst_parity,
        'balanced_worst_delta_count_tv': gate_tv,
        'balanced_worst_abs_delta_parity_error': gate_parity,
        'passes_gate': bool(gate_tv <= TV_GATE and gate_parity <= PARITY_GATE),
        'max_formula_count_error': None if not formula_errors else float(max(formula_errors)),
        'signal_abs_delta_parity': float(max(abs(row['exact_delta_parity']) for row in rows)),
        'signal_abs_raw_parity': float(max(abs(row['exact_raw_parity']) for row in rows)),
        'rows': rows,
    }


def combine_port_gates(default, crossed):
    """Holdout pass/fail uses the predeclared default port and the crossed port."""
    def largest(key):
        values = [record[key] for record in (default, crossed) if record.get(key) is not None]
        return None if not values else float(max(values))
    return {
        'default_port_passes_gate': bool(default['passes_gate']),
        'crossed_port_passes_gate': bool(crossed['passes_gate']),
        'worst_delta_count_tv': largest('worst_delta_count_tv'),
        'worst_abs_delta_parity_error': largest('worst_abs_delta_parity_error'),
        'balanced_worst_delta_count_tv': largest('balanced_worst_delta_count_tv'),
        'balanced_worst_abs_delta_parity_error': largest('balanced_worst_abs_delta_parity_error'),
        'passes_gate': bool(default['passes_gate'] and crossed['passes_gate']),
        'max_formula_count_error': largest('max_formula_count_error'),
        'signal_abs_delta_parity': largest('signal_abs_delta_parity'),
        'signal_abs_raw_parity': largest('signal_abs_raw_parity'),
    }


def onsite_sector_atom(model, parameters):
    """P_sector|g> for an onsite product, with ||g||_Fock = 1. Not renormalized."""
    amplitudes, phases = parameters[:model.L], parameters[model.L:]
    return np.exp(model.occ @ (amplitudes + 1j * phases) - 0.5 * np.logaddexp(0., 2 * amplitudes).sum())


def onsite_parity_alpha_lower(model, state, samples=48, seed=17):
    """Search lower bound on sup |<v|U†(-1)^{N_I}U|w>| over onsite sector atoms.

    A search maximum is only a lower bound on alpha. It can prove that
    B^2 alpha already exceeds the signal. It is not a certified supremum.
    """
    rng = np.random.default_rng(seed)
    density = np.clip(np.abs(state)**2 @ model.occ, 1e-5, 1 - 1e-5)
    center = 0.5 * np.log(density / (1.0 - density))
    atoms = [onsite_sector_atom(model, np.r_[center, np.zeros(model.L)])]
    while len(atoms) < samples:
        atoms.append(onsite_sector_atom(model, np.r_[
            center + rng.normal(0.0, 1.2, model.L),
            rng.uniform(-math.pi, math.pi, model.L)]))
    best = 0.0
    best_where = None
    signals = []
    for holes, particles in (default_port(model.L), crossed_port(model.L)):
        for phi in GATE_PHASES:
            readout = interferometer_readout(model, state, holes, particles, phi, math.pi / 4)
            signals.append({
                'holes': list(holes), 'particles': list(particles), 'phi': float(phi),
                'abs_raw_parity': abs(readout['raw_parity']),
                'abs_delta_parity': abs(readout['delta_parity']),
                'abs_T': abs(complex(readout['T_real'], readout['T_imag']))})
            for left_index, left in enumerate(atoms):
                for right in atoms[left_index:]:
                    magnitude = abs(orbit_parity_element(model, left, right, holes, particles, phi))
                    if magnitude > best:
                        best = float(magnitude)
                        best_where = {'left': left_index, 'phi': float(phi),
                                      'holes': list(holes), 'particles': list(particles)}
    return {
        'operator': 'U† (-1)^{N_I} U',
        'atom_class': 'onsite product sector projections with full-Fock normalization 1',
        'alpha_lower': best,
        'alpha_is_search_lower_bound_not_a_supremum': True,
        'pair_count_per_setting': samples * (samples + 1) // 2,
        'samples': samples,
        'best_location': best_where,
        'signals': signals,
        'arithmetic': 'float64, no interval enclosure',
    }


def training_point(spec, moment):
    try:
        started = time.perf_counter()
        model = make_model(spec)
        state = normalized(model.evolve([moment])[0])
        port = default_port(spec['L'])
        approximations = fast_approximations(model, state, spec_seed(spec, moment, 'fast'))
        coherence = transfer_expectation(model, state, port[0], port[1])
        direction = approximations['single_direction']['fidelity']
        families = {}
        for name, record in approximations.items():
            readout = slim_readout(model, state, record['state'], [port], GATE_PHASES, (math.pi / 4,))
            families[name] = {
                'fidelity': record['fidelity'],
                'conditional_rank1_fidelity_upper': record.get('conditional_rank1_fidelity_upper'),
                'global_optimum_certified': False,
                **{key: readout[key] for key in readout if key != 'rows'},
                'rows': readout['rows']}
        return {'spec': spec, 'time': moment, 'dimension': model.dim,
                'score': float(abs(coherence) * (1.0 - direction)),
                'T_abs': float(abs(coherence)), 'T': [float(coherence.real), float(coherence.imag)],
                'single_direction_fidelity': direction, 'families': families,
                'seconds': time.perf_counter() - started, 'error': None}
    except Exception:
        return {'spec': spec, 'time': moment, 'error': traceback.format_exc()}


def holdout_fast(spec, moment):
    try:
        started = time.perf_counter()
        model = make_model(spec)
        state = normalized(model.evolve([moment])[0])
        ports = [default_port(spec['L']), crossed_port(spec['L'])]
        approximations = fast_approximations(model, state, spec_seed(spec, moment, 'holdout'))
        families = {}
        for name, record in approximations.items():
            gate = slim_readout(model, state, record['state'], ports, GATE_PHASES, (math.pi / 4,))
            survey = slim_readout(model, state, record['state'], ports, SURVEY_PHASES, SURVEY_ALPHAS)
            families[name] = {
                'fidelity': record['fidelity'], 'global_optimum_certified': False,
                'passes_gate': gate['passes_gate'],
                'balanced_worst_delta_count_tv': gate['balanced_worst_delta_count_tv'],
                'balanced_worst_abs_delta_parity_error': gate['balanced_worst_abs_delta_parity_error'],
                'survey_worst_delta_count_tv': survey['worst_delta_count_tv'],
                'survey_worst_abs_delta_parity_error': survey['worst_abs_delta_parity_error'],
                'max_formula_count_error': survey['max_formula_count_error'],
                'signal_scale': max(abs(row['exact_delta_parity']) for row in gate['rows'])}
        return {'spec': spec, 'time': moment, 'dimension': model.dim, 'families': families,
                'seconds': time.perf_counter() - started, 'error': None}
    except Exception:
        return {'spec': spec, 'time': moment, 'error': traceback.format_exc()}


def pin_gpu(device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    os.environ['WORKER_CUDA'] = 'cuda:0'
    from pairing_formation.projected_gaussian import torch_module
    torch = torch_module()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.cuda.set_device(0)


def gpu_probe():
    from pairing_formation.projected_gaussian import torch_module
    torch = torch_module()
    torch.zeros(8, device='cuda:0')
    return {'name': torch.cuda.get_device_name(0), 'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}


def _cross_level_fraction(matrix):
    if matrix is None or not np.isfinite(matrix).all():
        return None
    levels = matrix.shape[0] // 2
    cross = 0.0
    for i in range(levels):
        for j in range(levels):
            if i == j:
                continue
            block = matrix[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            cross += float(np.linalg.norm(block)**2)
    total = float(np.linalg.norm(matrix))
    return None if total < 1e-15 else math.sqrt(cross) / total


def decomposition_job(spec, moment, seed, count, full_starts, onsite_starts):
    """Adaptive symmetry-orbit columns. full_starts=0 removes the cross-level oracle."""
    try:
        from pairing_formation.columns import adaptive_columns
        started = time.perf_counter()
        device = os.environ.get('WORKER_CUDA', 'cpu')
        model = make_model(spec)
        state = normalized(model.evolve([moment])[0])
        history, atoms, _parameters, matrices, _coefficients = adaptive_columns(
            model, state, count=count, seed=seed, device=device,
            full_starts=full_starts, onsite_starts=onsite_starts)
        repair, coefficients, residual, _dual = repair_minimum_mass(atoms, state)
        gaussian_part = atoms @ coefficients
        repaired = gaussian_part + residual
        port = default_port(spec['L'])
        gaussian_metrics = slim_readout(
            model, state, normalized(gaussian_part), [port], GATE_PHASES, (math.pi / 4,))
        crossed_metrics = slim_readout(
            model, state, normalized(gaussian_part), [crossed_port(spec['L'])], GATE_PHASES, (math.pi / 4,))
        repaired_metrics = slim_readout(
            model, state, normalized(repaired), [port], GATE_PHASES, (math.pi / 4,))
        cross = [_cross_level_fraction(matrix) for matrix in matrices]
        finite_cross = [value for value in cross if value is not None]
        return {
            'spec': spec, 'time': moment, 'seed': seed, 'count': count, 'full_starts': full_starts,
            'device': os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu'),
            'least_squares_fidelity': history[-1]['fidelity'],
            'least_squares_mass': history[-1]['coefficient_mass'],
            'repair': {key: repair[key] for key in repair if key != 'finite_dual_max_column_overlap'},
            'fock_support_above_1e-8': int(np.sum(np.abs(residual) > 1e-8)),
            'gaussian_only_state_error': float(np.linalg.norm(state - gaussian_part)),
            'repaired_state_error': float(np.linalg.norm(state - repaired)),
            'fock_columns': int(sum(bool(row['fock_selected']) for row in history)),
            'mean_cross_level_fraction': None if not finite_cross else float(np.mean(finite_cross)),
            'max_cross_level_fraction': None if not finite_cross else float(np.max(finite_cross)),
            'gaussian_only_gate': {key: gaussian_metrics[key] for key in gaussian_metrics if key != 'rows'},
            'crossed_port_gate': {key: crossed_metrics[key] for key in crossed_metrics if key != 'rows'},
            'repaired_exact_reconstruction_gate': {
                key: repaired_metrics[key] for key in repaired_metrics if key != 'rows'},
            'repaired_state_matches_target_by_construction': True,
            'seconds': time.perf_counter() - started, 'error': None}
    except Exception:
        return {'spec': spec, 'time': moment, 'seed': seed, 'error': traceback.format_exc()}


def general_job(spec, moment, seed, starts):
    try:
        from pairing_formation.columns import onsite_oracle
        from pairing_formation.projected_gaussian import fit_general
        started = time.perf_counter()
        device = os.environ.get('WORKER_CUDA', 'cpu')
        model = make_model(spec)
        state = normalized(model.evolve([moment])[0])
        _atom, parameters, _report = onsite_oracle(model, state, starts=4, seed=seed)
        report, _matrix, projection = fit_general(
            model, state, parameters, starts=starts, seed=seed, device=device, maxiter=160)
        projection = np.asarray(projection, complex)
        renorm = normalized(projection)
        readout = slim_readout(
            model, state, renorm, [default_port(spec['L'])], GATE_PHASES, (math.pi / 4,))
        crossed = slim_readout(
            model, state, renorm, [crossed_port(spec['L'])], GATE_PHASES, (math.pi / 4,))
        return {
            'spec': spec, 'time': moment, 'seed': seed,
            'full_number_sector_fidelity': float(report['best_found_fidelity']),
            'seniority_zero_weight_in_number_sector': float(report['seniority_zero_probability']),
            'seniority_renormalized_fidelity': fidelity(state, renorm),
            'global_optimum_certified': False,
            'gate': {key: readout[key] for key in readout if key != 'rows'},
            'crossed_port_gate': {key: crossed[key] for key in crossed if key != 'rows'},
            'seconds': time.perf_counter() - started, 'error': None}
    except Exception:
        return {'spec': spec, 'time': moment, 'seed': seed, 'error': traceback.format_exc()}


def canonical_mass_job(spec, witness, batches, batch_size, seed, keep_atoms):
    """Monte Carlo B_2 = d_{n,2} E |<u|g>|^3. Valid only for the k=2 witness u."""
    try:
        from pairing_formation.projected_gaussian import torch_module
        from pairing_formation.gaussian_bins import haar_atoms
        started = time.perf_counter()
        torch = torch_module()
        device = os.environ.get('WORKER_CUDA', 'cpu')
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        model = make_model(spec)
        witness_tensor = torch.as_tensor(np.asarray(witness, complex), dtype=torch.complex128, device=device)
        modes = 2 * model.L
        frame = math.exp(log_frame_dimension(modes, 2))
        total = total_square = total_abs2 = 0.0
        seen = 0
        max_overlap = 0.0
        top_weights, top_atoms = [], []
        reservoir = []
        for batch_index in range(batches):
            atoms = haar_atoms(model, batch_size, device, torch, generator=generator)
            overlap = atoms @ witness_tensor.conj()
            magnitude3 = overlap.abs().pow(3)
            total += float(magnitude3.sum())
            total_square += float(magnitude3.square().sum())
            total_abs2 += float(overlap.abs().square().sum())
            max_overlap = max(max_overlap, float(overlap.abs().max()))
            seen += batch_size
            if keep_atoms:
                weights = magnitude3.detach().cpu().numpy()
                chosen = np.argpartition(weights, -16)[-16:]
                selected = atoms.index_select(0, torch.as_tensor(chosen, device=device)).detach().cpu().numpy()
                for weight, atom in zip(weights[chosen], selected):
                    top_weights.append(float(weight))
                    top_atoms.append(atom)
                if len(top_weights) > 32:
                    order = np.argpartition(np.asarray(top_weights), -32)[-32:]
                    top_weights = [top_weights[i] for i in order]
                    top_atoms = [top_atoms[i] for i in order]
                keys = (torch.log(torch.rand((batch_size,), device=device, generator=generator))
                        / magnitude3.clamp_min(1e-300)).detach().cpu().numpy()
                local = np.argpartition(keys, -16)[-16:]
                local_atoms = atoms.index_select(0, torch.as_tensor(local, device=device)).detach().cpu().numpy()
                for key, atom in zip(keys[local], local_atoms):
                    reservoir.append((float(key), atom))
                if len(reservoir) > 32:
                    reservoir = sorted(reservoir, key=lambda item: item[0])[-32:]
        mean = total / seen
        variance = max(0.0, total_square / seen - mean * mean)
        kept = None
        if keep_atoms:
            kept = {'top_weights': top_weights, 'top_atoms': np.asarray(top_atoms, complex),
                    'reservoir_atoms': np.asarray([atom for _key, atom in reservoir], complex)}
        return {
            'samples': seen, 'sum_magnitude3': total, 'sum_magnitude3_square': total_square,
            'sum_abs_overlap_squared': total_abs2, 'max_overlap': max_overlap,
            'mass_mean': frame * mean, 'mass_stderr': frame * math.sqrt(variance / seen),
            'mean_abs_overlap_squared': total_abs2 / seen, 'frame_d2': frame,
            'seconds': time.perf_counter() - started, 'kept': kept, 'error': None}
    except Exception:
        return {'error': traceback.format_exc()}


def bnb_box(amplitudes, region, seed):
    try:
        result = certified_onsite_overlap(
            np.asarray(amplitudes), tolerance=0.01, max_boxes=2500, starts=2, seed=seed, region=region)
        result['error'] = None
        return result
    except Exception:
        return {'error': traceback.format_exc()}


def root_region():
    boxes = tuple((0.0, math.pi / 2, 0.0, 2 * math.pi) for _ in range(4))
    return ((0.0, math.pi / 2, 0.0, 0.0),) + boxes[1:]


def subdivide(region, target):
    regions = [tuple(tuple(box) for box in region)]
    order = [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)]
    step = 0
    while len(regions) < target and step < 16:
        level, kind = order[step % len(order)]
        step += 1
        expanded = []
        for current in regions:
            box = list(current)
            left, right = (box[level][0], box[level][1]) if kind == 0 else (box[level][2], box[level][3])
            if right - left < 1e-8:
                expanded.append(current)
                continue
            mid = 0.5 * (left + right)
            first, second = list(box), list(box)
            if kind == 0:
                first[level] = (left, mid, box[level][2], box[level][3])
                second[level] = (mid, right, box[level][2], box[level][3])
            else:
                first[level] = (box[level][0], box[level][1], left, mid)
                second[level] = (box[level][0], box[level][1], mid, right)
            expanded.extend((tuple(first), tuple(second)))
        regions = expanded
    return regions


def combine_mass(parts):
    samples = sum(part['samples'] for part in parts)
    total = sum(part['sum_magnitude3'] for part in parts)
    total_square = sum(part['sum_magnitude3_square'] for part in parts)
    mean = total / samples
    variance = max(0.0, total_square / samples - mean * mean)
    frame = parts[0]['frame_d2']
    return {'samples': samples, 'mass_mean': frame * mean,
            'mass_stderr': frame * math.sqrt(variance / samples),
            'max_overlap': max(part['max_overlap'] for part in parts),
            'mean_abs_overlap_squared': sum(part['sum_abs_overlap_squared'] for part in parts) / samples,
            'frame_d2': frame}


def pool_repair(atoms, target, label):
    matrix = np.asarray(atoms, complex)
    if matrix.ndim != 2:
        raise ValueError(label)
    if matrix.shape[0] != len(target):
        matrix = matrix.T
    repair, coefficients, residual, _dual = repair_minimum_mass(matrix, target)
    fit = matrix @ coefficients
    return {'label': label, 'columns': int(matrix.shape[1]),
            'repair_mass': repair['mass'],
            'gaussian_coefficient_mass': repair['gaussian_coefficient_mass'],
            'fock_repair_mass': repair['fock_repair_mass'],
            'least_squares_fidelity': fidelity(target, fit) if np.linalg.norm(fit) else 0.0,
            'state_error_before_fock_repair': float(np.linalg.norm(target - fit)),
            'fock_support_above_1e-8': int(np.sum(np.abs(residual) > 1e-8)),
            'finite_dictionary_dual_is_not_full_gaussian': True}


def projector_distance(left, right):
    def orth(matrix):
        q, r = np.linalg.qr(matrix)
        keep = np.abs(np.diag(r)) > 1e-8 * max(1.0, np.max(np.abs(np.diag(r))))
        return q[:, keep] if np.any(keep) else q[:, :1]
    left_q, right_q = orth(left), orth(right)
    overlap = float(np.linalg.norm(left_q.conj().T @ right_q)**2)
    return float(left_q.shape[1] + right_q.shape[1] - 2 * overlap)


def raise_errors(records, label):
    failed = [record for record in records if record.get('error')]
    if failed:
        raise RuntimeError(f'{label} failed:\n' + failed[0]['error'])


def choose_family(families):
    for name in FAMILY_ORDER:
        record = families.get(name)
        if record and record.get('passes_gate'):
            return name
    return 'adaptive_gaussian'


def blank_table():
    rows = []
    for modes in (8, 12, 16, 20, 24, 28):
        mass = float(blank_coefficient_mass(modes, 2))
        rows.append({
            'physical_modes': modes,
            'B2_blank': mass,
            'B2_blank_squared': mass * mass,
            'minimal_k_for_mass_1_1': minimal_k_within_mass(modes, 1.1),
            'canonical_mean_atom_fidelity_k2': canonical_mean_fidelity(modes, 2),
            'true_gaussian_atomic_mass': 1.0,
            'log_B2': log_blank_coefficient_mass(modes, 2)})
    identities = []
    for modes in (4, 8, 12, 16):
        identities.append({'modes': modes,
                           'log_d2': log_frame_dimension(modes, 2),
                           'log_binomial_half': math.log(math.comb(2 * modes, modes) / 2)})
    return {'rows': rows, 'frame_dimension_matches_binomial_half': identities,
            'interpretation': 'B2_blank is the k=2 coefficient mass of one normalized Gaussian. The atomic mass is 1.'}


def survey_task(spec, moment, ports, families):
    try:
        model = make_model(spec)
        state = normalized(model.evolve([moment])[0])
        approximations = {'exact': state}
        approximations.update({name: record['state'] for name, record in fast_approximations(
            model, state, spec_seed(spec, moment, 'survey')).items()})
        # `families` carries already chosen vectors for adaptive/general, keyed by name.
        for name, vector in families.items():
            approximations[name] = normalized(vector)
        output = []
        exact = approximations.pop('exact')
        for name, vector in approximations.items():
            readout = slim_readout(model, exact, vector, ports, SURVEY_PHASES, SURVEY_ALPHAS)
            output.append({'family': name,
                           'worst_delta_count_tv': readout['worst_delta_count_tv'],
                           'worst_abs_delta_parity_error': readout['worst_abs_delta_parity_error'],
                           'passes_balanced_rows_inside_gate': readout['passes_gate'],
                           'max_formula_count_error': readout['max_formula_count_error'],
                           'n_settings': len(readout['rows'])})
        return {'rows': output, 'error': None}
    except Exception:
        return {'error': traceback.format_exc()}


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        'gaussian_synthesis/experiments/pairing_formation/results/low_cost_interferometer_01')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    ctx = multiprocessing.get_context('spawn')
    protocol = {
        'training': {'levels': TRAIN_LEVELS, 'couplings': TRAIN_COUPLINGS, 'times': TRAIN_TIMES,
                     'spectrum': 'quartic_tilt = sorted(x+0.45 x^4+0.18 x^3), x picket coordinate, rescaled to [-1,1]'},
        'selection_score': '|T| * (1 - single_direction_fidelity) on ports (0,1)->(L-2,L-1), phi unused',
        'family_order_first_that_passes': FAMILY_ORDER,
        'gate': {'balanced_delta_count_tv': TV_GATE, 'balanced_abs_delta_parity_error': PARITY_GATE,
                 'phases': GATE_PHASES, 'alpha': math.pi / 4},
        'mass_ratio_target': MASS_RATIO_TARGET,
        'holdout_rule': 'edge_bunched L=6,8,10; quartic_tilt L=8,10; angular relative strengths '
                        '0.2,0.37,0.55,0.73 at L=6,8,10. Same selected G and t. Not used to choose the family.',
        'previously_used_conditions_not_relabelled_as_blind': True,
        'cpu_workers': CPU_WORKERS, 'gpu_workers': GPU_COUNT,
        'haar_samples_per_state_target': GPU_COUNT * HAAR_BATCHES * HAAR_BATCH,
        'adaptive_columns_training': ADAPTIVE_COUNT,
        'gate_uses_gaussian_columns_without_fock_repair': True,
        'fock_repaired_state_is_an_exact_reconstruction_not_a_readout_success': True}
    dump(output / 'protocol.json', protocol)
    print(f'CPUs scheduled {CPU_WORKERS}, GPUs {GPU_COUNT}', flush=True)
    cpu = ProcessPoolExecutor(max_workers=CPU_WORKERS, mp_context=ctx)
    gpu_pools = [ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=pin_gpu, initargs=(device,))
                 for device in range(GPU_COUNT)]
    try:
        probes = [pool.submit(gpu_probe) for pool in gpu_pools]
        gpu_names = [future.result() for future in probes]
        print('GPUs', gpu_names, flush=True)
        dump(output / 'hardware.json', {'cpu_workers': CPU_WORKERS, 'affinity': len(os.sched_getaffinity(0)),
                                        'gpus': gpu_names})
        calibration = blank_table()
        dump(output / 'calibration_blank.json', calibration)

        grid_futures = [cpu.submit(training_point, quartic_spec(TRAIN_LEVELS, coupling), moment)
                        for coupling in TRAIN_COUPLINGS for moment in TRAIN_TIMES]
        grid = []
        for future in as_completed(grid_futures):
            grid.append(future.result())
            print(f'training {len(grid)}/{len(grid_futures)}', flush=True)
        raise_errors(grid, 'training grid')
        formula_error = max(family['max_formula_count_error'] or 0.0
                            for row in grid for family in row['families'].values())
        if formula_error > 1e-8:
            raise RuntimeError(f'interferometer formula failed on the training grid: {formula_error}')
        winner = max(grid, key=lambda row: (row['score'], row['T_abs']))
        print('selected', winner['spec'], winner['time'], 'score', winner['score'], flush=True)
        dump(output / 'training_grid.json', {'points': grid, 'selected': {'spec': winner['spec'], 'time': winner['time'],
                                                                            'score': winner['score']}})

        families = dict(winner['families'])
        # Holdouts and the cost study do not depend on the final family label.
        holdout_specs = []
        coupling, moment = winner['spec']['G'], winner['time']
        for levels in (6, 8, 10):
            holdout_specs.append(edge_spec(levels, coupling))
        for levels in (8, 10):
            holdout_specs.append(quartic_spec(levels, coupling))
        for levels in (6, 8, 10):
            for relative in CHANNEL_STRENGTHS:
                holdout_specs.append(angular_spec(levels, coupling, relative))
        holdout_futures = [cpu.submit(holdout_fast, spec, moment) for spec in holdout_specs]

        train_spec = winner['spec']
        sibling = quartic_spec(4, coupling)
        spectator = {'kind': 'spectator', 'active': train_spec, 'gap': 4.0}
        fermi_model = make_model(train_spec)
        sibling_model = make_model(sibling)
        spectator_model = make_model(spectator)
        train_state = normalized(fermi_model.evolve([moment])[0])
        fermi_state = normalized(fermi_model.initial)
        sibling_state = normalized(sibling_model.evolve([moment])[0])
        spectator_state = normalized(embed_active_state(fermi_model, spectator_model, train_state))
        replica_reports = {}
        witnesses = {}
        for label, model, state in (
                ('training', fermi_model, train_state),
                ('fermi', fermi_model, fermi_state),
                ('sibling_L4', sibling_model, sibling_state),
                ('spectator', spectator_model, spectator_state)):
            witness, report = Replica2(model).solve(state, maxiter=400)
            replica_reports[label] = {key: report[key] for key in (
                'k', 'physical_modes', 'q', 'residual', 'moment_dual_extent_lower', 'optimizer_success', 'iterations')}
            replica_reports[label]['witness_alignment_with_state'] = float(
                abs(np.vdot(witness, state)) / (np.linalg.norm(witness) * np.linalg.norm(state)))
            witnesses[label] = witness
            print('replica', label, 'residual', report['residual'], flush=True)

        def submit_gpu(index, function, *args):
            return gpu_pools[index % GPU_COUNT].submit(function, *args)

        gpu_futures = []
        mass_plans = {
            'fermi': (train_spec, witnesses['fermi'], False),
            'training': (train_spec, witnesses['training'], True),
            'sibling_L4': (sibling, witnesses['sibling_L4'], False),
            'spectator': (spectator, witnesses['spectator'], False)}
        task_index = 0
        mass_future_keys = []
        for label, (spec, witness, keep) in mass_plans.items():
            for device in range(GPU_COUNT):
                gpu_futures.append(submit_gpu(
                    task_index, canonical_mass_job, spec, witness, HAAR_BATCHES, HAAR_BATCH,
                    spec_seed(label, device, 'haar'), keep))
                mass_future_keys.append(label)
                task_index += 1

        adaptive_keys = []
        for seed_index in range(GPU_COUNT):
            gpu_futures.append(submit_gpu(
                task_index, decomposition_job, train_spec, moment,
                1000 + seed_index, ADAPTIVE_COUNT, 2, 8))
            adaptive_keys.append(('train_full', seed_index))
            task_index += 1
        gpu_futures.append(submit_gpu(task_index, decomposition_job, sibling, moment, 1100, 12, 2, 8))
        adaptive_keys.append(('sibling_full', 0))
        task_index += 1
        for offset, spec in enumerate(holdout_specs):
            gpu_futures.append(submit_gpu(
                task_index, decomposition_job, spec, moment, 2000 + offset, HOLDOUT_COUNT, 1, 4))
            adaptive_keys.append(('holdout', offset))
            task_index += 1
            gpu_futures.append(submit_gpu(
                task_index, general_job, spec, moment, 3000 + offset, 2))
            adaptive_keys.append(('holdout_general', offset))
            task_index += 1
        gpu_futures.append(submit_gpu(task_index, general_job, train_spec, moment, 4100, 4))
        adaptive_keys.append(('train_general', 0))
        task_index += 1
        gpu_futures.append(submit_gpu(task_index, general_job, sibling, moment, 4101, 4))
        adaptive_keys.append(('sibling_general', 0))

        onsite_futures = [cpu.submit(decomposition_job, train_spec, moment, 5000 + seed, ADAPTIVE_COUNT, 0, 8)
                          for seed in range(32)]
        magic_train, magic_train_vector = matched_magic_witness(fermi_model, train_state, starts=8, seed=11)
        magic_sibling, magic_sibling_vector = matched_magic_witness(sibling_model, sibling_state, starts=8, seed=12)
        dicke = normalized(np.ones(sibling_model.dim, complex))
        magic_dicke, magic_dicke_vector = matched_magic_witness(sibling_model, dicke, starts=6, seed=13)
        regions = subdivide(root_region(), 128)
        bnb_jobs = []
        for label, vector in (('sibling_state', sibling_state), ('sibling_magic', magic_sibling_vector),
                              ('dicke_magic', magic_dicke_vector)):
            for index, region in enumerate(regions):
                bnb_jobs.append((label, cpu.submit(bnb_box, vector, region, 7000 + index)))
        print(f'submitted gpu {len(gpu_futures)}, onsite {len(onsite_futures)}, bnb {len(bnb_jobs)}', flush=True)

        gpu_results = []
        for done, future in enumerate(as_completed(gpu_futures), start=1):
            gpu_results.append(future.result())
            if done % 8 == 0 or done == len(gpu_futures):
                print(f'gpu {done}/{len(gpu_futures)}', flush=True)
        # as_completed loses the key order. Re-get in submission order.
        gpu_results = [future.result() for future in gpu_futures]
        raise_errors(gpu_results, 'gpu')
        cursor = 0
        mass_groups = {label: [] for label in ('fermi', 'training', 'sibling_L4', 'spectator')}
        for label in mass_future_keys:
            mass_groups[label].append(gpu_results[cursor])
            cursor += 1
        adaptive_results = []
        for key in adaptive_keys:
            adaptive_results.append((key, gpu_results[cursor]))
            cursor += 1
        if cursor != len(gpu_results):
            raise RuntimeError(f'gpu result cursor {cursor} != {len(gpu_results)}')

        onsite_results = [future.result() for future in onsite_futures]
        raise_errors(onsite_results, 'onsite adaptive')
        bnb_grouped = {label: [] for label in ('sibling_state', 'sibling_magic', 'dicke_magic')}
        for label, future in bnb_jobs:
            result = future.result()
            if result.get('error'):
                raise RuntimeError(result['error'])
            bnb_grouped[label].append(result)
        holdouts = [future.result() for future in holdout_futures]
        raise_errors(holdouts, 'holdouts')

        def bnb_summary(parts):
            upper = max(part['overlap_upper'] for part in parts)
            lower = max(part['overlap_lower'] for part in parts)
            return {'overlap_lower': lower, 'overlap_upper': upper, 'gap': upper - lower,
                    'closed': bool(upper <= lower + 0.01),
                    'boxes_processed': int(sum(part['boxes_processed'] for part in parts)),
                    'workers': len(parts), 'global_upper_bound': True,
                    'arithmetic': 'float64 interval bounds, no directed rounding'}

        certificates = {label: bnb_summary(parts) for label, parts in bnb_grouped.items()}
        masses = {label: combine_mass(parts) for label, parts in mass_groups.items()}
        atom_parts = [part for part in mass_groups['training'] if part.get('kept')]
        top_atoms, top_weights = [], []
        for part in atom_parts:
            kept = part['kept']
            top_atoms.extend(list(kept['top_atoms']))
            top_weights.extend(list(kept['top_weights']))
        if top_atoms:
            order = np.argsort(np.asarray(top_weights))[-32:]
            canonical_pool = pool_repair(np.asarray(top_atoms, complex)[order].T, train_state, 'canonical_top_weight')
            per_device = [part['kept']['reservoir_atoms'][:8] for part in atom_parts]
            reservoir_matrix = np.concatenate(per_device, axis=0)
            uniform_pool = pool_repair(reservoir_matrix.T, train_state, 'canonical_weighted_reservoir')
        else:
            canonical_pool = uniform_pool = None

        train_full = [record for key, record in adaptive_results if key[0] == 'train_full']
        best_train = min(train_full, key=lambda record: record['repair']['mass'])
        train_general = next(record for key, record in adaptive_results if key[0] == 'train_general')
        families['general_projected'] = {
            'fidelity': train_general['seniority_renormalized_fidelity'],
            'full_number_sector_fidelity': train_general['full_number_sector_fidelity'],
            'seniority_zero_weight_in_number_sector': train_general['seniority_zero_weight_in_number_sector'],
            'global_optimum_certified': False,
            **train_general['gate']}
        families['adaptive_gaussian'] = {
            'fidelity': 1.0 - best_train['gaussian_only_state_error']**2 / 2,  # placeholder replaced below
            'repair_mass': best_train['repair']['mass'],
            'gaussian_coefficient_mass': best_train['repair']['gaussian_coefficient_mass'],
            'gaussian_only_state_error': best_train['gaussian_only_state_error'],
            'global_optimum_certified': False,
            **best_train['gaussian_only_gate']}
        families['adaptive_gaussian']['fidelity'] = None
        frozen = choose_family(families)
        mass_lower_magic = math.sqrt(max(magic_train['extent_lower_numerical'], 0.0))
        mass_lower = max(1.0, mass_lower_magic)
        mass_upper = best_train['repair']['mass']
        ratio = mass_upper / mass_lower
        # A witness phase makes Re<y|psi> equal to the modulus without changing Gaussian overlaps.
        witness_overlap = float(abs(np.vdot(magic_sibling_vector, sibling_state)))
        sibling_full = next(record for key, record in adaptive_results if key[0] == 'sibling_full')
        sibling_repair = sibling_full['repair']['mass']
        sibling_error = sibling_full['repaired_state_error']
        outside = outside_configuration_fraction(
            witness_overlap, float(np.linalg.norm(magic_sibling_vector)), sibling_error,
            sibling_repair, certificates['sibling_magic']['overlap_upper'])
        dicke_overlap = float(abs(np.vdot(magic_dicke_vector, dicke)))
        dicke_atom = np.full(6, 0.25, complex)
        dicke_mass = float(abs(np.linalg.lstsq(dicke_atom[:, None], dicke, rcond=None)[0][0]))
        dicke_outside = outside_configuration_fraction(
            dicke_overlap, float(np.linalg.norm(magic_dicke_vector)), 0.0, dicke_mass,
            certificates['dicke_magic']['overlap_upper'])

        # Spectator constructive invariance from the best training decomposition is not stored as atoms.
        # Recompute one onsite-only embedding check with the analytic Fermi sea and the interacting embed.
        spectator_overlap = abs(np.vdot(
            spectator_state, normalized(embed_active_state(fermi_model, spectator_model, train_state))))
        blank_active = float(blank_coefficient_mass(2 * train_spec['L'], 2))
        blank_spectator = float(blank_coefficient_mass(2 * spectator_model.L, 2))

        fermi_exact = float(blank_coefficient_mass(2 * train_spec['L'], 2))
        fermi_mc = masses['fermi']['mass_mean']
        alpha = onsite_parity_alpha_lower(fermi_model, train_state)
        signal_abs = max(row['abs_raw_parity'] for row in alpha['signals'])
        delta_abs = max(row['abs_delta_parity'] for row in alpha['signals'])
        budget_alpha = mass_upper * mass_upper * alpha['alpha_lower']
        best_onsite = min(onsite_results, key=lambda record: record['repair']['mass'])
        cost = {
            'replica': replica_reports,
            'canonical_mass_monte_carlo': masses,
            'fermi_blank_exact_B2': fermi_exact,
            'fermi_monte_carlo_over_exact': fermi_mc / fermi_exact,
            'canonical_monte_carlo_is_not_an_interval_certificate': True,
            'best_training_adaptive': best_train,
            'training_adaptive_masses': [record['repair']['mass'] for record in train_full],
            'training_adaptive_cross_level': [record['mean_cross_level_fraction'] for record in train_full],
            'onsite_only_masses': [record['repair']['mass'] for record in onsite_results],
            'onsite_only_best_mass': best_onsite['repair']['mass'],
            'onsite_only_best_gaussian_state_error': best_onsite['gaussian_only_state_error'],
            'onsite_only_smallest_gaussian_state_error': min(
                record['gaussian_only_state_error'] for record in onsite_results),
            'onsite_only_best_gaussian_coefficient_mass': best_onsite['repair']['gaussian_coefficient_mass'],
            'onsite_only_best_fock_repair_mass': best_onsite['repair']['fock_repair_mass'],
            'onsite_only_best_cross_level': best_onsite['mean_cross_level_fraction'],
            'canonical_top_pool': canonical_pool,
            'canonical_weighted_reservoir_pool': uniform_pool,
            'magic_training': magic_train,
            'magic_mass_lower': mass_lower_magic,
            'trivial_mass_lower': trivial_mass_lower_bound(train_state),
            'mass_upper': mass_upper,
            'mass_lower_used_for_ratio': mass_lower,
            'certified_mass_ratio': ratio,
            'mass_ratio_target': MASS_RATIO_TARGET,
            'mass_ratio_target_met': bool(ratio <= MASS_RATIO_TARGET),
            'blank_B2_active_modes': blank_active,
            'blank_B2_with_two_spectator_levels': blank_spectator,
            'spectator_embedded_state_overlap': float(spectator_overlap),
            'certificates': certificates,
            'sibling_outside_onsite_fraction': outside,
            'dicke_one_atom_mass': dicke_mass,
            'dicke_outside_onsite_fraction': dicke_outside,
            'dicke_certificate_must_not_be_positive': True,
            'moment_dual_extent_lower_training': replica_reports['training']['moment_dual_extent_lower'],
            'moment_dual_extent_lower_spectator': replica_reports['spectator']['moment_dual_extent_lower'],
            'magic_witness_norms': {
                'training': float(np.linalg.norm(magic_train_vector)),
                'sibling': float(np.linalg.norm(magic_sibling_vector)),
                'dicke': float(np.linalg.norm(magic_dicke_vector))},
            'sibling_adaptive': {
                'repair_mass': sibling_full['repair']['mass'],
                'gaussian_coefficient_mass': sibling_full['repair']['gaussian_coefficient_mass'],
                'fock_repair_mass': sibling_full['repair']['fock_repair_mass'],
                'gaussian_only_state_error': sibling_full['gaussian_only_state_error'],
                'repaired_state_error': sibling_full['repaired_state_error'],
                'mean_cross_level_fraction': sibling_full['mean_cross_level_fraction'],
                'least_squares_fidelity': sibling_full['least_squares_fidelity'],
                'global_optimum_certified': False},
            'interferometer_alpha_lower_bound': {
                **alpha,
                'budget': mass_upper,
                'budget_squared_times_alpha_lower': budget_alpha,
                'max_abs_raw_parity': signal_abs,
                'max_abs_delta_parity': delta_abs,
                'bound_already_exceeds_raw_parity': bool(budget_alpha >= signal_abs),
                'bound_already_exceeds_delta_parity': bool(budget_alpha >= delta_abs),
                'interpretation': 'B^2 alpha_lower already exceeds the observed parity signals, so the onsite-class inequality does not force mass outside that class.'}}
        dump(output / 'cost.json', cost)

        # Attach GPU controls onto holdouts.
        holdout_by_spec = {json.dumps(record['spec'], sort_keys=True): record for record in holdouts}
        for offset, spec in enumerate(holdout_specs):
            record = holdout_by_spec[json.dumps(spec, sort_keys=True)]
            adaptive = next(item for key, item in adaptive_results if key == ('holdout', offset))
            general = next(item for key, item in adaptive_results if key == ('holdout_general', offset))
            record['families']['adaptive_gaussian'] = {
                'gaussian_only_state_error': adaptive['gaussian_only_state_error'],
                'repair_mass': adaptive['repair']['mass'],
                'gaussian_coefficient_mass': adaptive['repair']['gaussian_coefficient_mass'],
                'fock_repair_mass': adaptive['repair']['fock_repair_mass'],
                'mean_cross_level_fraction': adaptive['mean_cross_level_fraction'],
                'global_optimum_certified': False,
                **combine_port_gates(adaptive['gaussian_only_gate'], adaptive['crossed_port_gate'])}
            record['families']['general_projected'] = {
                'fidelity': general['seniority_renormalized_fidelity'],
                'full_number_sector_fidelity': general['full_number_sector_fidelity'],
                'seniority_zero_weight_in_number_sector': general['seniority_zero_weight_in_number_sector'],
                'global_optimum_certified': False,
                **combine_port_gates(general['gate'], general['crossed_port_gate'])}
        frozen_holdout_failures = []
        for record in holdouts:
            chosen = record['families'].get(frozen)
            if chosen is None or not chosen.get('passes_gate'):
                frozen_holdout_failures.append({'spec': record['spec'], 'family_record': chosen})
        dump(output / 'holdouts.json', {'time': moment, 'coupling': coupling, 'frozen_family': frozen,
                                        'cases': holdouts, 'frozen_gate_failures': len(frozen_holdout_failures)})

        summary = {
            'selected_spec': train_spec, 'selected_time': moment, 'selected_score': winner['score'],
            'frozen_family': frozen,
            'frozen_family_gate': families.get(frozen),
            'training_families': {name: {key: value for key, value in record.items() if key != 'rows'}
                                  for name, record in families.items()},
            'certified_mass_ratio': ratio,
            'mass_ratio_target_met': bool(ratio <= MASS_RATIO_TARGET),
            'mass_upper': mass_upper, 'mass_lower': mass_lower,
            'fermi_monte_carlo_over_exact': cost['fermi_monte_carlo_over_exact'],
            'sibling_outside_onsite_fraction': outside,
            'dicke_outside_onsite_fraction': dicke_outside,
            'sibling_certificate_closed': certificates['sibling_magic']['closed'],
            'holdout_count': len(holdouts),
            'frozen_holdout_gate_failures': len(frozen_holdout_failures),
            'onsite_only_best_mass': cost['onsite_only_best_mass'],
            'full_oracle_best_mass': mass_upper,
            'seconds': time.perf_counter() - started}
        dump(output / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        plot(output, calibration, grid, cost, holdouts, frozen)
    finally:
        cpu.shutdown(wait=True, cancel_futures=False)
        for pool in gpu_pools:
            pool.shutdown(wait=True, cancel_futures=False)


def plot(output, calibration, grid, cost, holdouts, frozen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = [row['physical_modes'] for row in calibration['rows']]
    masses = [row['B2_blank'] for row in calibration['rows']]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.semilogy(modes, masses, 'o-', label='k=2 blank mass of one Gaussian')
    axis.axhline(1.0, color='0.3', ls='--', label='atomic mass')
    axis.set_xlabel('physical modes')
    axis.set_ylabel('coefficient mass')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'blank_mass.pdf')
    figure.savefig(output / 'blank_mass.png', dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    for coupling in TRAIN_COUPLINGS:
        rows = sorted((row for row in grid if row['spec']['G'] == coupling), key=lambda row: row['time'])
        axis.plot([row['time'] for row in rows], [row['score'] for row in rows], 'o-', label=f'G={coupling:g}')
    axis.set_xlabel('time')
    axis.set_ylabel('|T| times single-direction infidelity')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'training_score.pdf')
    figure.savefig(output / 'training_score.png', dpi=140)
    plt.close(figure)

    labels, errors = [], []
    for record in holdouts:
        chosen = record['families'].get(frozen, {})
        labels.append(record['spec']['kind'] + str(record['spec']['L']))
        errors.append(chosen.get('balanced_worst_abs_delta_parity_error', np.nan))
    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    axis.bar(range(len(errors)), errors)
    axis.axhline(PARITY_GATE, color='0.3', ls='--')
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=75, ha='right', fontsize=7)
    axis.set_ylabel('worst |delta parity| error')
    figure.tight_layout()
    figure.savefig(output / 'holdout_parity.pdf')
    figure.savefig(output / 'holdout_parity.png', dpi=140)
    plt.close(figure)


if __name__ == '__main__':
    main()
