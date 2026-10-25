#!/usr/bin/env python3
"""Archive sixth-round mechanism controls and the projected-Gaussian bound."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def row(record):
    return next(x for x in record['rows'] if x['elapsed'] == 1.)


def base_name(name):
    return name.split('_seed')[0]


def aggregate(data):
    results = []
    for family in sorted({base_name(r['family']) for r in data}):
        rows = [row(r) for r in data if base_name(r['family']) == family]
        results.append({'family': family, 'family_condition_seed_records': len(rows),
                        **{metric: float(np.mean([abs(x[metric]) if metric == 'negativity_error' else x[metric] for x in rows]))
                           for metric in ('fidelity', 'count_TV', 'energy_resolved_cross_RMS', 'negativity_error', 'unnormalized_filtered_trace_distance')},
                        'signed_negativity_error': float(np.mean([x['negativity_error'] for x in rows])),
                        **{component: {'signed_negativity_error': float(np.mean([x[component]['negativity_error'] for x in rows])),
                                       'absolute_negativity_error': float(np.mean([abs(x[component]['negativity_error']) for x in rows])),
                                       'unnormalized_filtered_trace_distance': float(np.mean([x[component]['unnormalized_filtered_trace_distance'] for x in rows]))}
                           for component in ('initial_state_only', 'dynamics_only')}})
    return results


def save(fig, output, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'{name}.{suffix}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]; results = root/'results'
    folders = {'known': 'orbit_response_known_predictions_01', 'unseen': 'orbit_response_unseen_predictions_01', 'swaps': 'orbit_swaps_predictions_01'}
    data = {name: read(results/folder/'evaluation.json') for name, folder in folders.items()}
    selection = read(results/'orbit_response_training_01/selection.json')
    fits = [read(p) for p in sorted((results/'orbit_response_training_01').glob('*_weight*_seed*.json'))]
    starts = [t for f in fits for b in f['blocks'] for t in b['restarts']]
    selected_best = [min(b['restarts'], key=lambda x: x['loss']) for f in selection['selected'] for b in f['blocks'] if b['restarts']]
    hashes = []
    for folder in ('orbit_response_freeze_01', 'orbit_swaps_freeze_01'):
        manifest = read(results/folder/'manifest.json')
        for record in manifest['records']:
            path = results/folder/record['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
            hashes.append(str(path.relative_to(root)))
    manifest = read(results/'orbit_response_freeze_01/manifest.json')
    assert hashlib.sha256((results/'orbit_response_training_01/selection.json').read_bytes()).hexdigest() == manifest['training_selection_sha256']
    previous = read(results/'reversal_refinement_predictions_01/evaluation.json')
    replay_error = 0.
    for record in data['known']:
        if not record['family'].startswith(('target_seed', 'witness_seed')):
            continue
        old_name = record['family'].replace('_seed', '_two_orbits_seed')
        old = next(r for r in previous if r['family'] == old_name and r['G'] == record['G'] and r['kick'] == record['kick'])
        for a, b in zip(record['rows'], old['rows']):
            for key in ('fidelity', 'count_TV', 'negativity_error', 'unnormalized_filtered_trace_distance'):
                replay_error = max(replay_error, abs(a[key]-b[key]))
    assert replay_error < 1e-10
    all_predictions = [r for batch in data.values() for r in batch]
    validations = read(results/'orbit_controls_validation_01/validation.json')
    random_checks = read(results/'projected_count_bound_01/random_checks.json')
    physics = read(results/'projected_count_bound_01/physics_certificates.json')
    echo = read(results/'count_echo_01/evaluation.json')
    aggregates = {k: aggregate(v) for k, v in data.items()}
    pairs = []
    for grid in ('known', 'unseen'):
        lookup = {(r['family'], r['case']): row(r) for r in data[grid]}
        for case in sorted({r['case'] for r in data[grid]}):
            for seed in (20261111, 20261112):
                record = {'grid': grid, 'case': case, 'seed': seed}
                for method in ('target', 'witness', 'response_hamiltonian', 'response_generators'):
                    r = lookup[f'{method}_seed{seed}', case]
                    record[method] = {'absolute_negativity_error': abs(r['negativity_error']),
                                      'unnormalized_filtered_trace_distance': r['unnormalized_filtered_trace_distance'],
                                      'initial_negativity_error': r['initial_state_only']['negativity_error'],
                                      'dynamics_negativity_error': r['dynamics_only']['negativity_error']}
                pairs.append(record)
    audit = {'scope': 'The earlier narrow witness advantage is largely initial-bias cancellation, not improved variational dynamics. A separate analytic orbital-invariant counting bound and finite-angle coherence certificate are established here; novelty and canonical-specific prediction remain unestablished.',
             'prediction_counts': {k: len(v) for k, v in data.items()},
             'new_physical_conditions': 8, 'known_physical_conditions': 6,
             'exact_own_initial_state_controls': len(all_predictions),
             'training_candidates': len(fits), 'reused_direct_candidates': sum(f.get('reused_direct_training', False) for f in fits),
             'optimizer_starts': len(starts), 'converged_starts': sum(x['success'] for x in starts),
             'selected_best_blocks': len(selected_best), 'selected_best_blocks_converged': sum(x['success'] for x in selected_best),
             'training_selection': [{k:r[k] for k in ('name','weight','initial_fidelity','training_tangent_response','allowed_weights')} for r in selection['selected']],
             'frozen_files_verified': len(hashes), 'verified_files': hashes,
             'training_selection_hash_verified': True, 'old_prediction_replay_max_metric_difference': replay_error,
             'aggregate': aggregates, 'paired_comparisons': pairs,
             'max_norm_error': max(r['max_norm_error'] for r in all_predictions),
             'max_energy_drift': max(r['integration'].get('max_energy_drift', 0.) for r in all_predictions),
             'integration_validation': validations,
             'max_refinement_state_difference': max(r['max_state_difference'] for r in validations),
             'max_refinement_negativity_difference': max(r['max_negativity_difference'] for r in validations),
             'general_projected_Gaussian_proof': 'PROJECTED_GAUSSIAN_COUNT_BOUND.md; full particle-number sector, arbitrary orbital basis; numerical bounds evaluated in float64, not interval arithmetic',
             'random_checks': random_checks,
             'random_case_count': sum(r['cases'] for r in random_checks['records']),
             'independent_full_pfaffian_checks': 3*len(random_checks['records']),
             'retrospective_physics': [{'L':d['L'], 'G':d['G'],
                                       'max_echo_row': max(d['rows'], key=lambda r:r['finite_angle_witness_excess']),
                                       'max_infidelity_row': max(d['rows'], key=lambda r:r['fidelity_certificate']['infidelity_lower_bound'])} for d in physics],
             'synthetic_echo_cases': len(echo['records']), 'synthetic_echo_certifications': sum(r['certified_outside_projected_Gaussian_convex_hull'] for r in echo['records']),
             'free_echo_controls': sum(r['kind']=='free_Fock_mixture' for r in echo['records']),
             'free_echo_false_positives': sum(r['certified_outside_projected_Gaussian_convex_hull'] for r in echo['records'] if r['kind']=='free_Fock_mixture'),
             'synthetic_echo_protocol': echo['protocol']}
    run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root/'tests'), '-v'], capture_output=True, text=True)
    log = run.stdout+run.stderr; (args.output/'tests.log').write_text(log)
    matched = re.search(r'Ran \d+ tests in [\d.]+s', log)
    audit['tests'] = {'returncode': run.returncode, 'summary': matched.group(0) if matched else 'No test summary'}
    if run.returncode:
        raise RuntimeError('Tests failed; see tests.log')
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2))

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    families = ['target', 'witness', 'response_hamiltonian', 'response_generators']
    labels = ['Direct', 'Witness', 'H response', 'Generator response']
    for ax, grid, title in zip(axes, ('known', 'unseen'), ('Six earlier reversal conditions', 'Eight unseen spectrum / reversal conditions')):
        rows = [next(r for r in aggregates[grid] if r['family']==f) for f in families]
        x = np.arange(len(rows))
        ax.bar(x-.17, [r['initial_state_only']['signed_negativity_error'] for r in rows], .34, color=colors[0], label='Initial-state error, exact propagation')
        ax.bar(x+.17, [r['dynamics_only']['signed_negativity_error'] for r in rows], .34, color=colors[1], label='Dynamical error from own initial state')
        ax.plot(x, [r['signed_negativity_error'] for r in rows], 'kD', label='Total signed error')
        ax.axhline(0, color='black', lw=.7)
        ax.set(xticks=x, xticklabels=labels, ylabel='Signed filtered negativity error at elapsed time 1', title=title)
        ax.tick_params(axis='x', labelsize=8); ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=7.5)
    fig.suptitle('The witness gain mainly cancels a dynamical bias\nMeans over physical conditions and two optimizer seeds; these are not independent seed trials')
    save(fig, args.output, 'initial_bias_control')

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    lookup = {r['family']: r for r in aggregates['swaps']}
    for i, (base, donor) in enumerate((('target','witness'),('witness','target'))):
        ax.plot(range(1, 6), [lookup[f'{base}_with_{donor}_m{m}']['negativity_error'] for m in range(1, 6)], 'o-', color=colors[i], label=f'{base.capitalize()} with one {donor} sector')
        ax.axhline(lookup[base]['negativity_error'], color=colors[i], ls='--', label=f'All {base}')
    ax.set(xticks=range(1, 6), xlabel='Replaced transfer-count sector m', ylabel='Mean absolute negativity error',
           title='The m=1 profile explains most of the gain\nInitial state changes with the swap; exact initial-state controls are retained')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, args.output, 'sector_swaps')

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True)
    for i, L in enumerate((8, 12, 16)):
        d = next(d for d in physics if d['L']==L and d['G']==4.)
        times = np.array([r['time'] for r in d['rows']])
        axes[0,0].plot(times, [r['fano'] for r in d['rows']], color=colors[i], label=f'L={L}')
        axes[0,1].plot(times, [.75**2*r['pure_echo_difference']-.2**2*r['mean'] for r in d['rows']], color=colors[i], label=f'L={L}')
        values = [r['fidelity_certificate']['infidelity_lower_bound'] for r in d['rows']]
        axes[1,0].semilogy(times, np.where(np.array(values)>0, values, np.nan), color=colors[i], label=f'L={L}')
    axes[0,0].axhline(1, color='black', ls='--', lw=1)
    axes[0,1].axhline(0, color='black', ls='--', lw=1)
    axes[0,0].set(ylabel='Var(m) / mean(m)', title='G=4: pure-state projected-Gaussian bound')
    axes[0,1].set(ylabel='K(0) - K(0.2) - 0.04 mean(m)', title='Finite-angle exclusion after 25% count dephasing')
    axes[1,0].set(ylabel='Global infidelity lower bound (float64)', title='All orbital bases; full number-sector normalization')
    for ax in (axes[0,0], axes[0,1], axes[1,0]):
        ax.set(xlabel='Original quench time'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    rows = [r for r in echo['records'] if r['L']==16 and r['G']==4. and r['time']==3.]
    x = np.arange(len(rows)); estimates = [r['estimate'] for r in rows]; radii = [r['simultaneous_one_sided_radius'] for r in rows]
    axes[1,1].errorbar(x, estimates, yerr=np.array([radii, np.zeros(len(rows))]), fmt='o', color=colors[2], capsize=4, label='Synthetic estimate / lower confidence bound')
    axes[1,1].plot(x, [r['finite_angle_excess'] for r in rows], 'x', color='black', label='Exact model')
    axes[1,1].axhline(0, color='black', ls='--', lw=1)
    axes[1,1].set(xticks=x, xticklabels=['0','0.1','0.25','0.5','1','Fock mix'], xlabel='Complete count-dephasing probability / free control',
                  ylabel='Finite-angle witness excess', title='L=16, G=4, t=3; synthetic measurements only')
    axes[1,1].legend(fontsize=7.3); axes[1,1].grid(alpha=.2)
    fig.suptitle('A stronger certificate from previously saved trajectories\nAnalytic scope is universal; these curves are retrospective and do not establish canonical prediction advantage')
    save(fig, args.output, 'projected_gaussian_certificate')
    print(json.dumps({k:audit[k] for k in ('prediction_counts','frozen_files_verified','tests','random_case_count','synthetic_echo_certifications','max_refinement_state_difference','old_prediction_replay_max_metric_difference')}))
