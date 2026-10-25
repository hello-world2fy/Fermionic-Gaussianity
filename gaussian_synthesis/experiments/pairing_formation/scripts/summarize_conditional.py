#!/usr/bin/env python3
"""Fixed third-round archive, including independent test output."""
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
    fits = [read(p) for p in sorted((result/'conditional_orbitals_01').glob('*.json'))]
    cc = [read(p) for p in sorted((result/'radial_cc_L12_G2_01').glob('*.json'))]
    dynamic = read(result/'conditional_dynamics_predictions_01/evaluation.json')
    static = read(result/'conditional_predictions_01/evaluation.json')
    scale = {L: read(result/f'conditional_scale_L{L}_predictions_01/evaluation.json') for L in (8, 16)}
    validation = read(result/'conditional_dynamics_validation_01/validation.json')+read(result/'conditional_scale_validation_01/validation.json')
    frozen_count = 0
    for directory in ['conditional_freeze_01', 'conditional_dynamics_freeze_01', 'conditional_scale_L8_freeze_01', 'conditional_scale_L16_freeze_01']:
        manifest = read(result/directory/'manifest.json')
        for item in manifest['records']:
            assert hashlib.sha256((result/directory/item['file']).read_bytes()).hexdigest() == item['sha256']
            frozen_count += 1
    aggregate = []
    for family in sorted(set(x['family'] for x in dynamic)):
        for signs, couplings in [('positive', [1.5, 2.5]), ('interaction_reversed', [-1.8])]:
            for t in [0., .25, .5, 1., 1.5, 2.]:
                rows = [y for x in dynamic if x['family'] == family and x['G'] in couplings for y in x['rows'] if y['elapsed'] == t]
                aggregate.append({'family': family, 'couplings': signs, 'elapsed': t, 'cases': len(rows),
                    **{metric: {'mean': float(np.mean([y[metric] for y in rows])), 'max': float(max(y[metric] for y in rows))}
                       for metric in ['fidelity', 'count_TV', 'energy_resolved_cross_RMS', 'occupation_max_error']}})
    all_dynamic = dynamic+scale[8]+scale[16]
    audit = {'scope': 'Conditional orbital mechanism and stronger baselines; canonical uniqueness remains unproven',
        'conditional_fits': [{k: x[k] for k in ['time', 'source', 'seed', 'fidelity', 'blocks']} for x in fits],
        'radial_cc': [{k: x[k] for k in ['time', 'rank', 'seed', 'best_found_fidelity', 'restarts']} for x in cc],
        'static_predictions': len(static), 'dynamic_predictions': len(dynamic), 'scale_predictions': len(scale[8])+len(scale[16]),
        'frozen_files_verified': frozen_count, 'dynamic_aggregate': aggregate,
        'max_dynamic_norm_error': max(x['max_norm_error'] for x in all_dynamic),
        'max_dynamic_energy_drift': max(x['integration'].get('max_energy_drift', 0.) for x in all_dynamic),
        'refinement_checks': validation, 'max_refinement_state_difference': max(x['max_state_difference'] for x in validation)}
    run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root/'tests'), '-v'], capture_output=True, text=True)
    log = run.stdout+run.stderr
    (args.output/'tests.log').write_text(log)
    audit['tests'] = {'returncode': run.returncode, 'summary': re.search(r'Ran \d+ tests in [\d.]+s', log).group(0)}
    if run.returncode:
        raise RuntimeError('Independent tests failed; see tests.log')
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2))
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for ax, t in zip(axes, [1., 2.15]):
        x = np.load(result/f'conditional_orbitals_01/target_t{t:g}_seed20261031.npz')['parameters']
        energies = 2*(np.arange(6)+6+.5)/12-1
        for m in range(1, 6):
            amplitude = np.exp(2*(x[m, :6]-max(x[m, :6])))
            amplitude /= amplitude.sum()
            ax.plot(energies, amplitude, 'o-', label=f'm={m}')
        ax.set(xlabel='Particle energy level', ylabel='Normalized orbital |z_j(m)|^2', title=f'Conditional fitted orbital, t={t:g}')
        ax.legend(); ax.grid(alpha=.2)
    for suffix in ['png', 'pdf']:
        fig.savefig(args.output/f'conditional_profiles.{suffix}')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    names = [('target_common', 'moving common orbital (17)'), ('target_conditioned', 'moving conditional (57)'),
             ('witness_conditioned', 'witness conditional (57)'), ('hamiltonian_R32', 'H response static (32)'),
             ('linear_rpa_q2', 'RPA q=2 static (92)'), ('linear_rpa_q3', 'RPA q=3 static (408)')]
    for name, label in names:
        rows = [x for x in aggregate if x['family'] == name and x['couplings'] == 'positive']
        for ax, metric in zip(axes, ['fidelity', 'count_TV', 'energy_resolved_cross_RMS']):
            ax.plot([x['elapsed'] for x in rows], [x[metric]['mean'] for x in rows], '.--' if name == 'witness_conditioned' else '.-', label=label)
    for ax, title in zip(axes, ['Fidelity', 'Count total variation', 'Energy-resolved pair RMS error']):
        ax.set(xlabel='Time after freeze / kick', ylabel=title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7.5)
    fig.suptitle('New holdouts G=1.5, 2.5; spectral kick 0, +/-0.3\nNonlinear coordinate counts and linear subspace dimensions are not equal computation costs')
    for suffix in ['png', 'pdf']:
        fig.savefig(args.output/f'conditional_predictions.{suffix}')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, G in zip(axes, [1.5, 2.5]):
        for family, label in [('target_common', 'common orbital'), ('target_conditioned', 'conditional orbitals')]:
            values = []
            for L in [8, 12, 16]:
                data = dynamic if L == 12 else scale[L]
                record = next(x for x in data if x['family'] == family and x['G'] == G and x['kick'] == 'none')
                values.append(next(x['count_TV'] for x in record['rows'] if x['elapsed'] == 1.))
            ax.plot([8, 12, 16], values, 'o-', label=label)
        ax.set(xlabel='Number of pair levels L', ylabel='Count TV at elapsed time 1', title=f'After G=2 to G={G:g} quench')
        ax.legend(); ax.grid(alpha=.2)
    for suffix in ['png', 'pdf']:
        fig.savefig(args.output/f'conditional_size_check.{suffix}')
    plt.close(fig)
    print(json.dumps({k: audit[k] for k in ['static_predictions', 'dynamic_predictions', 'scale_predictions', 'frozen_files_verified', 'tests', 'max_refinement_state_difference']}))
