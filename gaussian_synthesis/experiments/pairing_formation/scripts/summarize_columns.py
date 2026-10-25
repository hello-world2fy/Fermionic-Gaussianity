#!/usr/bin/env python3
"""Archive the second-round evidence; fixed inputs, new output directory only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence


def read(path):
    return json.loads(path.read_text())


def aggregate(evaluations):
    groups = {}
    for item in evaluations:
        name = item['family']
        if '_seed' in name:
            name = name.split('_seed')[0]+'_'+name.rsplit('_', 1)[1]
        key = name+'_'+item['transport']
        for row in item['rows']:
            groups.setdefault((key, row['elapsed']), []).append(row)
    result = []
    for (key, elapsed), rows in groups.items():
        record = {'family_transport': key, 'elapsed': elapsed, 'cases': len(rows)}
        for metric in ('fidelity', 'count_TV', 'cross_error', 'mean_error', 'variance_error'):
            values = np.array([r[metric] for r in rows])
            record[metric] = {'min': float(values.min()), 'mean': float(values.mean()),
                              'max': float(values.max()), 'mean_absolute': float(abs(values).mean())}
        result.append(record)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    results = root/'results'
    adaptive = [read(p) for p in sorted((results/'adaptive_full_L12_G2_01').glob('*.json'))]
    costs = [read(p) for p in sorted((results/'column_costs_01').glob('*.json'))]
    response_training = [read(p) for p in sorted((results/'response_columns_01').glob('*.json'))]
    frozen = read(results/'column_freeze_01/manifest.json')
    response_frozen = read(results/'response_freeze_01/manifest.json')
    first_predictions = read(results/'column_predictions_01/evaluation.json')
    response_predictions = read(results/'response_predictions_01/evaluation.json')
    sectors = read(results/'sector_diagnostics_01/summary.json')
    stats = aggregate(response_predictions)
    audit = {'scope': 'Second-round benchmark and mechanism screening; novelty success gate remains false',
             'adaptive': [{k: v for k, v in x.items() if k != 'history'} | {'final': x['history'][-1]} for x in adaptive],
             'costs': costs, 'response_training': [{k: v for k, v in x.items() if k != 'history'} |
                                                  {'final': x['history'][-1]} for x in response_training],
             'first_frozen_predictions': len(first_predictions), 'response_frozen_predictions': len(response_predictions),
             'prediction_max_norm_error': max(x['max_norm_error'] for x in first_predictions+response_predictions),
             'response_aggregate': stats, 'sector_cases': len(sectors),
             'sector_max_continuity_error': max(x['peak']['continuity_error'] for x in sectors),
             'sector_max_count_preservation_error': max(x['initial_control_count_error'] for x in sectors),
             'sector_max_twin_collective_cross_error': max(x['time_reversal_mixture_initial_cross_error'] for x in sectors),
             'response_frozen_records': response_frozen['records']}
    verified = 0
    for directory, manifest in [('column_freeze_01', frozen), ('response_freeze_01', response_frozen)]:
        for record in manifest['records']:
            assert hashlib.sha256((results/directory/record['file']).read_bytes()).hexdigest() == record['sha256']
            verified += 1
    audit['frozen_sha256_verified'] = verified
    model = picket_fence(12, 2.)
    repairs = []
    for report in costs:
        name = f"t{report['time']:g}_seed{report['seed']}"
        training = np.load(results/f'adaptive_full_L12_G2_01/{name}.npz')
        repair = np.load(results/f'column_costs_01/{name}.npz')
        error = np.linalg.norm(training['atoms']@repair['coefficients']+repair['fock_repair']-training['target'])
        repairs.append(float(error))
    audit['max_cost_reconstruction_residual'] = max(repairs)
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2))

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for t, color in [(.5, 'C0'), (1., 'C1'), (2.15, 'C2')]:
        for seed in (20261027, 20261028):
            data = read(results/f'adaptive_full_L12_G2_01/t{t:g}_seed{seed}.json')
            history = data['history']
            axes[0].semilogy([x['columns'] for x in history], [max(1-x['fidelity'], 1e-16) for x in history],
                             color=color, alpha=.8, label=f't={t:g}' if seed == 20261027 else None)
    axes[0].set(xlabel='Frozen Gaussian symmetry orbits R', ylabel='1 - reconstruction fidelity', title='Local adaptive full-Gaussian oracle')
    for family, label in [('canonical', 'k=2 canonical pool'), ('husimi', 'Husimi pool'), ('haar', 'Haar pool')]:
        h = read(results/f'column_freeze_01/{family}_pool_selection.json')
        axes[1].plot([x['columns'] for x in h], [x['fidelity'] for x in h], label=label)
    h = read(results/'adaptive_full_L12_G2_01/t1_seed20261027.json')['history']
    axes[1].plot([x['columns'] for x in h], [x['fidelity'] for x in h], color='k', label='adaptive continuum oracle')
    axes[1].set(xlabel='Selected orbits R', ylabel='Reconstruction fidelity at t=1', ylim=(0, 1.025), title='Finite sample pools vs adaptive search')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'columns.{suffix}')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    families = [('state_only_R32_static', 'state only, 32'), ('hamiltonian_R32_static', 'H response, 32'),
                ('canonical_response_R32_static', 'canonical response, 32'),
                ('linear_rpa_q2_static', 'RPA q=2, dim 92'), ('linear_rpa_q3_static', 'RPA q=3, dim 408')]
    for name, label in families:
        rows = sorted([x for x in stats if x['family_transport'] == name], key=lambda x: x['elapsed'])
        for ax, metric in zip(axes, ['fidelity', 'count_TV', 'cross_error']):
            ax.plot([x['elapsed'] for x in rows], [x[metric]['mean'] for x in rows], marker='.', label=label)
    for ax, title in zip(axes, ['Fidelity', 'Count total variation', 'Collective four-point error']):
        ax.set(xlabel='Time after freeze / kick', ylabel=title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('All fresh holdouts averaged: G=1.6, 2.6; phase 0, +/- pi/5; both seeds\nStatic spans; dimensions differ as shown')
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'response_predictions.{suffix}')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for L in (8, 12, 16):
        case = read(results/f'sector_diagnostics_01/L{L}_G2_picket.json')
        axes[0].plot(np.arange(L//2), case['peak']['current'], 'o-', label=f'L={L}')
    axes[0].axhline(0, color='k', lw=.7)
    axes[0].set(xlabel='Transfer sector m', ylabel='J(m -> m+1)', title='Net current zero at first mean pulse')
    for spectrum, label in [('picket', 'dispersive levels'), ('two_flat_bands', 'one direction per band')]:
        case = read(results/f'sector_diagnostics_01/L12_G2_{spectrum}.json')
        axes[1].plot(np.arange(6), case['peak']['current'], 'o-', label=label)
    axes[1].axhline(0, color='k', lw=.7)
    axes[1].set(xlabel='Transfer sector m', ylabel='J(m -> m+1)', title='Counterflow does not prove new channels')
    case = read(results/'sector_diagnostics_01/L12_G2_picket.json')
    for name, label in [('exact', 'original pure state'), ('time_reversal_mixture', 'time-reversal mixture'),
                        ('phase_alignment', 'aligned sector phases'), ('conditional_rank_one', 'rank one per sector')]:
        rows = case['curves'][name]
        axes[2].plot([x['elapsed'] for x in rows], [x['variance'] for x in rows], 'o-', label=label)
    axes[2].set(xlabel='Time after first mean pulse', ylabel='Variance of transferred pairs', title='All controls start with identical p(m)')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'sector_phase_controls.{suffix}')
    plt.close(fig)
    print(json.dumps({key: audit[key] for key in ('first_frozen_predictions', 'response_frozen_predictions',
          'prediction_max_norm_error', 'frozen_sha256_verified', 'max_cost_reconstruction_residual', 'sector_cases')}))
