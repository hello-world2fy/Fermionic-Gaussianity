#!/usr/bin/env python3
"""Retrospective transfer readout on the original L12 G2 pairing problem.

Saved fits are selected by their original state fidelity, never by the new
observable. They are representation controls, not forecasts. Stochastic
trajectories start at the Fermi sea and use no later exact states.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import json
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.pairing_channels import PairKernelModel
from pairing_formation.fitting import excitation_features
from pairing_formation.kernel_stochastic import torch_kernel_run, numpy_kernel_rhs
from pairing_formation.stochastic import numpy_rhs, initial_samples
from pairing_formation.multipair_noise import distinct_transfer_square_symbol, transfer_operators
from scripts.pilot_pair_symbols import digest


TIMES = [0., .5, 1., 2.15]


def model_and_checks():
    original = picket_fence(12, 2.)
    model = PairKernelModel(original.energies, np.full((12, 12), original.g))
    delta = model.H-original.H
    h_error = float(np.max(abs(delta.data), initial=0.))
    s = initial_samples(12, 8, 'gaussian', np.random.default_rng(2210))
    rhs_error = float(np.max(abs(numpy_rhs(s, original.energies, original.g)
                                  -numpy_kernel_rhs(s, model.energies, model.kernel))))
    assert h_error < 1e-13 and rhs_error < 1e-13
    return model, {'uniform_H_error': h_error, 'uniform_RHS_error': rhs_error}


def snapshot_controls(out):
    out.mkdir(parents=True, exist_ok=False)
    root = out.parent
    model, checks = model_and_checks()
    X = transfer_operators(model)[0]
    hashes = {}
    rows = []
    selected = []
    saved_states = {}

    def load_json(path):
        hashes[str(path)] = digest(path)
        return json.loads(path.read_text())

    def best_saved(paths, score, state_key):
        candidates = [(path, load_json(path)) for path in sorted(paths)]
        assert candidates
        path, report = max(candidates, key=lambda item: item[1][score])
        data_path = path.with_suffix('.npz')
        hashes[str(data_path)] = digest(data_path)
        return np.load(data_path)[state_key], report[score], str(data_path)

    for time in TIMES[1:]:
        reference_path = root/f'replica2_L12_G2_01/t{time:g}.npz'
        hashes[str(reference_path)] = digest(reference_path)
        exact = np.load(reference_path)['target']
        exact = exact/np.linalg.norm(exact)
        fresh = model.evolve([time])[0]
        checks[f'exact_state_repeat_t{time:g}'] = float(np.linalg.norm(fresh-exact))
        assert np.linalg.norm(fresh-exact) < 1e-11
        states = {'exact': exact}
        for family in ('pbcs', 'radial', 'radial_slope'):
            path = root/f'key_fits_L12_G2_01/{family}_t{time:g}.json'
            report = load_json(path)
            features = excitation_features(model)
            if family == 'radial_slope':
                features = np.column_stack((features, features*model.m[:, None]/model.r))
            d = features.shape[1]
            pars = np.array(report['fit']['parameters'])
            raw = features@(pars[:d]+1j*pars[d:])
            groups = [np.arange(model.dim)] if family == 'pbcs' else [np.flatnonzero(model.m == m) for m in range(model.r+1)]
            v = np.zeros(model.dim, complex)
            for ids in groups:
                local = np.exp(raw[ids]-max(raw[ids].real))
                local /= np.linalg.norm(local)
                v[ids] = local*np.vdot(local, exact[ids])
            v /= np.linalg.norm(v)
            states[family] = v
            mismatch = abs(abs(np.vdot(exact, v))**2-report['fit']['fidelity'])
            assert mismatch < 1e-10
            selected.append({'time': time, 'method': family, 'file': str(path), 'fidelity_reconstruction_error': float(mismatch)})

        specs = [(f'pair_CC_rank{rank}', root/'cc_L12_G2_01', f't{time:g}_rank{rank}.json', 'best_found_fidelity', 'state') for rank in (1, 2, 6)]
        specs += [
            ('radial_full_CC', root/'radial_cc_L12_G2_01', f't{time:g}_rank6_seed*.json', 'best_found_fidelity', 'state'),
            ('conditioned_orbitals', root/'conditional_orbitals_01', f'target_t{time:g}_seed*.json', 'fidelity', 'approximation'),
            ('two_body_Jastrow', root/'jastrow_fits_01', f'two_body_t{time:g}_seed*.json', 'fidelity', 'approximation'),
            ('conditioned_Jastrow', root/'jastrow_fits_extended_01', f'conditional_two_body_t{time:g}_seed*.json', 'fidelity', 'approximation'),
            ('two_conditioned_orbits', root/('multiorbit_training_refined_01' if time == 1 else 'multiorbit_fits_01'), f'target_t{time:g}_seed*.json', 'fidelity', 'approximation')]
        for method, directory, pattern, score, key in specs:
            v, fidelity, path = best_saved(directory.glob(pattern), score, key)
            v = v/np.linalg.norm(v)
            mismatch = abs(abs(np.vdot(exact, v))**2-fidelity)
            assert mismatch < 1e-9
            states[method] = v
            selected.append({'time': time, 'method': method, 'file': path, 'fidelity_reconstruction_error': float(mismatch)})

        exact_coefficient = None
        exact_pulses = {}
        for method, state in states.items():
            saved_states[f'{method}_t{time:g}'] = state
            second = np.vdot(state, X@X@state)
            coefficient = -2*np.array([second.real, second.imag])
            if method == 'exact':
                exact_coefficient = coefficient
            even = np.column_stack([state*(model.m % 2 == p) for p in (0, 1)])
            diagonal = np.column_stack([state*(model.m == m) for m in range(model.r+1)])
            initial = np.column_stack((even, diagonal))
            pulse_rows = []
            for phase in (0., float(np.pi/4)):
                Q = np.exp(-1j*phase)*X+np.exp(1j*phase)*X.conj().T
                moved = expm_multiply(-.4j*Q, initial, traceA=0.)
                probs = [np.bincount(model.m, weights=np.sum(abs(a)**2, axis=1), minlength=model.r+1) for a in (moved[:, :2], moved[:, 2:])]
                m = np.arange(model.r+1)
                delta = float((probs[0]-probs[1])@(m*m)-(probs[0]@m)**2+(probs[1]@m)**2)
                if method == 'exact':
                    exact_pulses[phase] = (probs, delta)
                refp, refdelta = exact_pulses[phase]
                pulse_rows.append({'phase': phase, 'epsilon': .4, 'variance_difference': delta,
                    'variance_difference_error': abs(delta-refdelta),
                    'even_output_TV_error': float(.5*np.sum(abs(probs[0]-refp[0]))),
                    'diagonal_output_TV_error': float(.5*np.sum(abs(probs[1]-refp[1]))),
                    'coherence_deletion_count_TV': float(.5*np.sum(abs(probs[0]-probs[1]))),
                    'max_trace_error': float(max(abs(p.sum()-1) for p in probs))})
            rows.append({'time': time, 'method': method, 'fidelity': float(abs(np.vdot(exact, state))**2),
                'coefficient': coefficient.tolist(), 'coefficient_L2_error': float(np.linalg.norm(coefficient-exact_coefficient)),
                'count_TV_error': float(.5*np.sum(abs(np.bincount(model.m, weights=abs(state)**2-abs(exact)**2, minlength=model.r+1)))),
                'pulse_rows': pulse_rows})
    np.savez_compressed(out/'states.npz', **saved_states)
    report = {'classification': 'Retrospective new observable on saved fits, selected by prior fidelity. No forecasts, new canonical syntheses, or global fit bounds.',
        'L': 12, 'G': 2., 'times': TIMES[1:], 'q': 0, 'input_hashes': hashes, 'checks': checks,
        'selected': selected, 'rows': rows, 'states_sha256': digest(out/'states.npz'),
        'general_PG_scope': 'Prior general projected-Gaussian local best fits coincided with onsite PBCS; omitted here as a redundant numerical fit, not excluded globally.'}
    (out/'snapshots.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'checks': checks, 'pulse_rows': [{k: r[k] for k in ('method', 'fidelity', 'coefficient', 'coefficient_L2_error')} for r in rows if r['time'] == 2.15]}, indent=2), flush=True)


def stochastic_job(task):
    out, device, kind, seed, dt, count = task
    model, checks = model_and_checks()
    cache = []
    def extra(s, torch):
        if not cache:
            cache.append(torch.full((model.r, model.r), 1/model.r, dtype=torch.complex128, device=s.device))
        plus = s[:, model.r:, 0]+1j*s[:, model.r:, 1]
        minus = s[:, :model.r, 0]-1j*s[:, :model.r, 1]
        second = distinct_transfer_square_symbol(plus, minus, cache[0])
        return torch.stack((-2*second.real, -2*second.imag), dim=-1)
    report, blocks = torch_kernel_run(model, [], count=count, kind=kind, seed=seed, times=TIMES, dt=dt, device=device,
        extra_measure=extra, extra_names=['readout_real', 'readout_imag'])
    report['model_checks'] = checks
    tag = f'{kind}_seed{seed}_dt{dt:g}'
    np.savez_compressed(out/(tag+'.npz'), blocks=blocks, times=TIMES)
    (out/(tag+'.json')).write_text(json.dumps(report, indent=2))
    return {'kind': kind, 'seed': seed, 'dt': dt, 'samples': count, 'device': device,
        'file': tag+'.npz', 'sha256': digest(out/(tag+'.npz')),
        'report_file': tag+'.json', 'report_sha256': digest(out/(tag+'.json'))}


def stochastic(out, samples):
    assert (out/'snapshots.json').exists() and not (out/'stochastic_protocol.json').exists()
    seeds = [2211, 2212]
    protocol = {'classification': 'Known L12 G2 conditions; stochastic trajectories from Fermi sea, no fitting.',
        'times': TIMES, 'kinds': ['gaussian', 'discrete'], 'seeds': seeds, 'dt': [.004, .002],
        'independent_samples': 4*samples, 'samples_per_run': samples,
        'step_repeats': 'Both time steps reuse identical initial random samples; do not count them twice.'}
    (out/'stochastic_protocol.json').write_text(json.dumps(protocol, indent=2))
    tasks = [(out, device, kind, seed, dt, samples) for device, (kind, seed, dt) in enumerate(
        (kind, seed, dt) for kind in ('gaussian', 'discrete') for seed in seeds for dt in (.004, .002))]
    jobs = []
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(stochastic_job, task) for task in tasks]):
            jobs.append(future.result())
            print(json.dumps(jobs[-1]), flush=True)
    (out/'stochastic_jobs.json').write_text(json.dumps(jobs, indent=2))


def assess(out):
    assert not (out/'evaluation.json').exists()
    paths = [out/p for p in ('snapshots.json', 'stochastic_protocol.json', 'stochastic_jobs.json')]
    snapshot, protocol, jobs = [json.loads(p.read_text()) for p in paths]
    hashes = {str(p): digest(p) for p in paths}
    hashes.update(snapshot['input_hashes'])
    hashes[str(out/'states.npz')] = snapshot['states_sha256']
    blocks = {}
    diagnostics = []
    for job in jobs:
        for field, hfield in (('file', 'sha256'), ('report_file', 'report_sha256')):
            hashes[str(out/job[field])] = job[hfield]
        blocks[job['kind'], job['seed'], job['dt']] = np.load(out/job['file'])['blocks']
        diagnostics.append(json.loads((out/job['report_file']).read_text()))
    for path, value in hashes.items():
        assert digest(Path(path)) == value
    rows = []
    step_checks = []
    for kind in protocol['kinds']:
        for dt in protocol['dt']:
            combined = np.concatenate([blocks[kind, seed, dt] for seed in protocol['seeds']], axis=1)
            for ti, time in enumerate(TIMES):
                avg = combined[ti, :, 3:].mean(axis=0)
                se = combined[ti, :, 3:].std(axis=0, ddof=1)/np.sqrt(combined.shape[1])
                exact = np.array([0., 0.] if time == 0 else next(r['coefficient'] for r in snapshot['rows'] if r['method'] == 'exact' and r['time'] == time))
                rows.append({'method': kind, 'dt': dt, 'time': time, 'mean': avg.tolist(), 'estimated_SE': se.tolist(),
                    'exact': exact.tolist(), 'L2_error': float(np.linalg.norm(avg-exact)),
                    'error_over_estimated_SE': (abs(avg-exact)/se).tolist()})
        for seed in protocol['seeds']:
            delta = blocks[kind, seed, .004]-blocks[kind, seed, .002]
            step_checks.append({'kind': kind, 'seed': seed, 'max_mean_difference': float(np.max(abs(delta.mean(axis=1)))),
                'max_block_difference': float(np.max(abs(delta)))})
    result = {'classification': 'Original-model readout controls, retrospective snapshot capacities and stochastic propagations; no canonical-specific discovery or new-condition prediction.',
        'input_hashes': hashes, 'snapshot_rows': snapshot['rows'], 'stochastic_rows': rows, 'step_checks': step_checks,
        'independent_stochastic_samples': protocol['independent_samples'], 'stochastic_runs_with_step_repeats': len(jobs),
        'max_number_drift': max(r['max_number_drift'] for r in diagnostics),
        'max_spin_length_squared_drift': max(r['max_spin_length_squared_drift'] for r in diagnostics),
        'max_energy_symbol_drift': max(r['max_energy_symbol_drift'] for r in diagnostics),
        'MC_scope': 'Estimated standard errors; no rigorous confidence certificate. Polynomial spin symbols are not positive quantum density matrices.'}
    (out/'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'refined_stochastic': [r for r in rows if r['dt'] == .002], 'step_checks': step_checks}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['snapshots', 'stochastic', 'assess'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=131072)
    args = parser.parse_args()
    if args.stage == 'snapshots':
        snapshot_controls(args.output)
    elif args.stage == 'stochastic':
        stochastic(args.output, args.samples)
    else:
        assess(args.output)
