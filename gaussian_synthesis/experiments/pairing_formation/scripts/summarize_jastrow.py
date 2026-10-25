#!/usr/bin/env python3
"""Archive fourth-round evidence without changing previous result directories."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    result = root/'results'
    dynamic = read(result/'jastrow_predictions_01/evaluation.json')
    scale = {L: read(result/f'jastrow_scale_L{L}_predictions_01/evaluation.json') for L in (8, 16)}
    validation = read(result/'jastrow_validation_01/validation.json')+read(result/'jastrow_scale_validation_01/validation.json')
    fits = [read(p) for folder in ('jastrow_fits_01', 'jastrow_fits_extended_01') for p in sorted((result/folder).glob('*.json'))]
    cubes = read(result/'density_cubes_01/diagnostics.json')
    verified, shared_errors = 0, []
    for folder in ('jastrow_freeze_01', 'jastrow_scale_L8_freeze_01', 'jastrow_scale_L16_freeze_01'):
        manifest = read(result/folder/'manifest.json')
        for record in manifest['records']:
            assert hashlib.sha256((result/folder/record['file']).read_bytes()).hexdigest() == record['sha256']
            verified += 1
            if 'shared_initial_state_difference' in record:
                shared_errors.append(record['shared_initial_state_difference'])
    aggregate = []
    for family in sorted(set(x['family'] for x in dynamic)):
        for sign, couplings in [('positive', [1.45, 2.55]), ('interaction_reversed', [-1.7])]:
            for elapsed in (0., .25, .5, 1., 1.5, 2.):
                rows = [y for x in dynamic if x['family'] == family and x['G'] in couplings for y in x['rows'] if y['elapsed'] == elapsed]
                aggregate.append({'family': family, 'couplings': sign, 'elapsed': elapsed, 'cases': len(rows),
                    **{metric: {'mean': float(np.mean([y[metric] for y in rows])), 'min': float(min(y[metric] for y in rows)), 'max': float(max(y[metric] for y in rows))}
                       for metric in ('fidelity', 'count_TV', 'energy_resolved_cross_RMS', 'connected_occupation_RMS')}})
    all_predictions = dynamic+scale[8]+scale[16]
    starts = [trial for fit in fits for trial in fit['restarts']]
    audit = {'scope': 'Density-Jastrow alternatives, higher-order count dependence, and fit-independent probability constraints; novelty and canonical uniqueness remain unestablished',
             'fits': fits, 'L12_predictions': len(dynamic), 'size_predictions': len(scale[8])+len(scale[16]),
             'optimizer_starts': {'total': len(starts), 'reported_converged': sum(x['success'] for x in starts),
                                  'amplitude_bound_active': sum(x['amplitude_bound_active'] for x in starts)},
             'frozen_files_verified': verified, 'max_shared_initial_state_difference': max(shared_errors),
             'aggregate': aggregate, 'cube_diagnostics': cubes,
             'max_norm_error': max(x['max_norm_error'] for x in all_predictions),
             'max_energy_drift': max(x['integration'].get('max_energy_drift', 0.) for x in all_predictions),
             'validation': validation, 'max_refinement_state_difference': max(x['max_state_difference'] for x in validation),
             'max_refinement_count_difference': max(x['max_count_difference'] for x in validation)}
    run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root/'tests'), '-v'], capture_output=True, text=True)
    log = run.stdout+run.stderr
    (args.output/'tests.log').write_text(log)
    audit['tests'] = {'returncode': run.returncode, 'summary': re.search(r'Ran \d+ tests in [\d.]+s', log).group(0)}
    if run.returncode:
        raise RuntimeError('Tests failed; see tests.log')
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2))
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    names = [('shared_collective_0', 'Common orbital (17)'), ('shared_two_body', 'Two-density Jastrow (70)'),
             ('shared_collective_3', 'Cubic count orbitals (47)'), ('shared_conditioned', 'Conditioned orbitals (57)'),
             ('shared_conditional_two_body', 'Combined (100)'), ('linear_rpa_q3', 'Static RPA q=3 (408)')]
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    for i, (family, label) in enumerate(names):
        rows = [x for x in aggregate if x['family'] == family and x['couplings'] == 'positive']
        for ax, metric in zip(axes, ('fidelity', 'count_TV', 'energy_resolved_cross_RMS')):
            ax.plot([x['elapsed'] for x in rows], [x[metric]['mean'] for x in rows], 'o--' if family.startswith('linear') else 'o-', ms=3, color=colors[i], label=label)
    for ax, title in zip(axes, ('Fidelity', 'Count total variation', 'Energy-resolved pair RMS error')):
        ax.set(xlabel='Time after freeze / kick', ylabel=title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7.2, loc='lower left')
    fig.suptitle('Held-out G=1.45, 2.55; spectral kick 0, +/-0.35\nNonlinear families share the same initial state; RPA uses its own initial projection')
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'jastrow_predictions.{suffix}')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for i, (family, label) in enumerate(names[:-1]):
        rows = []
        for L in (8, 12, 16):
            data = dynamic if L == 12 else scale[L]
            record = next(x for x in data if x['family'] == family and x['G'] == 2.55 and x['kick'] == 'none')
            rows.append(next(x for x in record['rows'] if x['elapsed'] == 1.))
        for ax, metric in zip(axes, ('count_TV', 'energy_resolved_cross_RMS')):
            ax.plot([8, 12, 16], [x[metric] for x in rows], 'o-', color=colors[i], label=label.split(' (')[0])
    axes[0].set(ylabel='Count total variation'); axes[1].set(ylabel='Energy-resolved pair RMS error')
    for ax in axes:
        ax.set(xlabel='Number of pair levels L', xticks=[8, 12, 16]); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same-initial-state size check: G=2 to 2.55, elapsed time 1')
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'jastrow_size_check.{suffix}')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].semilogy([x['time'] for x in cubes], [x['infidelity_lower_bound_float64'] for x in cubes], 'o-')
    axes[0].set(xlabel='Original G=2 evolution time', ylabel='Analytic infidelity lower bound (float64)', title='Independent of local optimization')
    latest = cubes[-1]
    labels = ['Exact state', 'Two-density', 'Cubic count', 'Combined']
    values = [latest['best_cube']['invariant']]+[latest['controls'][name]['exact_selected_cube_invariant'] for name in ('two_body', 'collective_3', 'conditional_two_body')]
    axes[1].bar(labels, values, color=[colors[7], colors[1], colors[2], colors[4]])
    axes[1].set(ylabel='Normalized alternating amplitude product', title='Pulse cube selected after inspecting exact state')
    axes[1].tick_params(axis='x', labelsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'density_constraint.{suffix}')
    plt.close(fig)
    print(json.dumps({k: audit[k] for k in ('L12_predictions', 'size_predictions', 'frozen_files_verified', 'tests', 'max_refinement_state_difference', 'max_norm_error', 'max_energy_drift')}))
