#!/usr/bin/env python3
"""Audit the fifth-round fixed archives; no fitting or endpoint selection."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


METRICS = ('fidelity', 'count_TV', 'energy_resolved_cross_RMS',
           'negativity_error', 'unnormalized_filtered_trace_distance')


def primary(record):
    return next(row for row in record['rows'] if row['elapsed'] == 1.)


def metric(row, name):
    return abs(row[name]) if name == 'negativity_error' else row[name]


def aggregate(data):
    result = []
    for family in sorted({r['family'] for r in data}):
        for sign in ('positive', 'reversed'):
            rows = [primary(r) for r in data if r['family'] == family and (r['G'] > 0) == (sign == 'positive')]
            if not rows:
                continue
            result.append({'family': family, 'couplings': sign, 'cases': len(rows),
                           **{m: {'mean': float(np.mean([metric(r, m) for r in rows])),
                                  'min': float(min(metric(r, m) for r in rows)),
                                  'max': float(max(metric(r, m) for r in rows))} for m in METRICS}})
    return result


def optimizer_audit(paths):
    fits, starts, selected = [], [], []
    for path in paths:
        record = read(path)
        fits.append({'file': str(path), 'fidelity': record['fidelity']})
        for block in record['blocks']:
            trials = block['restarts']
            starts.extend(trials)
            if trials:
                best = max(trials, key=lambda x: x['fidelity'])
                selected.append({'file': str(path), 'm': block['m'], **best})
    return {'fits': fits, 'starts': len(starts), 'converged_starts': sum(x['success'] for x in starts),
            'selected_blocks': len(selected), 'converged_selected_blocks': sum(x['success'] for x in selected),
            'nonconverged_selected': [x for x in selected if not x['success']],
            'global_optimum_certified': False}


def comparison(data):
    rows = []
    lookup = {(r['family'], r['G'], r['kick']): primary(r) for r in data}
    for G, kick in sorted({(r['G'], r['kick']) for r in data}):
        for seed in (20261111, 20261112):
            direct = lookup[f'target_two_orbits_seed{seed}', G, kick]
            witness = lookup[f'witness_two_orbits_seed{seed}', G, kick]
            rows.append({'G': G, 'kick': kick, 'seed': seed,
                         **{m: {'direct': metric(direct, m), 'witness': metric(witness, m),
                                'witness_lower': bool(metric(witness, m) < metric(direct, m))}
                            for m in ('negativity_error', 'unnormalized_filtered_trace_distance')}})
    return rows


def save(fig, output, name):
    for ext in ('png', 'pdf'):
        fig.savefig(output/f'{name}.{ext}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    result = root/'results'
    folders = {
        'pilot': 'interband_predictions_01',
        'scale_L8': 'interband_scale_L8_predictions_01',
        'scale_L16': 'interband_scale_L16_predictions_01',
        'confirmation': 'reversal_confirmation_predictions_01',
        'refinement': 'reversal_refinement_predictions_01',
        'pilot_linear_controls': 'interband_linear_controls_01',
        'confirmation_linear_controls': 'reversal_linear_controls_01',
    }
    data = {key: read(result/folder/'evaluation.json') for key, folder in folders.items()}
    verified = []
    freeze_folders = ('interband_freeze_01', 'interband_scale_L8_freeze_01',
                      'interband_scale_L16_freeze_01', 'reversal_confirmation_freeze_01',
                      'reversal_refinement_freeze_01', 'entanglement_witness_freeze_01')
    for folder in freeze_folders:
        for record in read(result/folder/'manifest.json')['records']:
            path = result/folder/record['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
            verified.append(str(path.relative_to(root)))
    validation = read(result/'multiorbit_validation_01/validation.json')+read(result/'multiorbit_scale_validation_01/validation.json')
    witnesses = read(result/'entanglement_witness_predictions_01/evaluation.json')
    witness_summary = []
    for family in sorted({r['family'] for r in witnesses['records']}):
        records = [r for r in witnesses['records'] if r['family'] == family]
        eligible = [r for r in records if r['predicted_expectation'] < -1e-8]
        witness_summary.append({'family': family, 'cases': len(records), 'predicted_entanglement_cases': len(eligible),
                                'raw_certified': sum(r['entanglement_certified_in_synthetic_data'] for r in records),
                                'certified_with_predicted_entanglement': sum(r['entanglement_certified_in_synthetic_data'] for r in eligible)})
    diagnostics = [read(result/f'interband_diagnostics_01/L{L}_{spectrum}.json') for L in (8, 12, 16) for spectrum in ('picket', 'flat')]
    all_predictions = [r for key, records in data.items() if 'linear_controls' not in key for r in records]
    numerical_differences = []
    old = {r['name']: r for r in data['confirmation']}
    for new in data['refinement']:
        before, after = primary(old[new['name']]), primary(new)
        numerical_differences.append({'name': new['name'], **{m: abs(metric(after, m)-metric(before, m)) for m in METRICS}})
    audit = {
        'scope': 'Fixed-count interband entanglement and narrow witness-derived prediction advantage; important novel physics and canonical-specific mechanism remain unestablished',
        'prediction_counts': {key: len(records) for key, records in data.items()},
        'new_family_condition_predictions': sum(len(data[k]) for k in ('pilot', 'scale_L8', 'scale_L16', 'confirmation')),
        'counts_are_independent_physical_conditions': False,
        'canonical_profiles_are_actual_measure_samples': False,
        'frozen_files_verified': len(verified), 'verified_files': verified,
        'aggregate_at_elapsed1': {key: aggregate(records) for key, records in data.items()},
        'original_fits': optimizer_audit(sorted((result/'multiorbit_fits_01').glob('*.json'))),
        'refined_training_fits': optimizer_audit(sorted((result/'multiorbit_training_refined_01').glob('*.json'))),
        'scale_fits': optimizer_audit(sorted((result/'interband_scale_L8_freeze_01').glob('*_fit.json'))+sorted((result/'interband_scale_L16_freeze_01').glob('*_fit.json'))),
        'confirmation_comparison': comparison(data['confirmation']),
        'refined_comparison': comparison(data['refinement']),
        'training_refinement_changes_at_primary_time': numerical_differences,
        'max_norm_error': max(r['max_norm_error'] for r in all_predictions),
        'max_energy_drift': max(r['integration'].get('max_energy_drift', 0.) for r in all_predictions),
        'integration_validation': validation,
        'max_refinement_state_difference': max(x['max_state_difference'] for x in validation),
        'max_refinement_negativity_difference': max(x['max_filtered_negativity_difference'] for x in validation),
        'source_max_local_normal_norm': max(r['local_normal_norm'] for d in diagnostics for row in d['rows'] for r in row['sources']['records']),
        'source_max_rank_beyond_two_norm': max(r['rank_beyond_two_norm'] for d in diagnostics for row in d['rows'] for r in row['sources']['records']),
        'synthetic_witness_summary': witness_summary,
        'synthetic_witness_familywise_failure_probability': witnesses['familywise_failure_probability'],
        'synthetic_witness_note': 'Single-orbit predictions have a degenerate zero PT eigenspace. Raw certifications from its arbitrary numerical eigenvector are retained but are not predictions of nonzero entanglement.',
        'local_dephasing': read(result/'interband_diagnostics_01/schmidt_dephasing.json'),
    }
    run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root/'tests'), '-v'], capture_output=True, text=True)
    log = run.stdout+run.stderr
    (args.output/'tests.log').write_text(log)
    summary = re.search(r'Ran \d+ tests in [\d.]+s', log)
    audit['tests'] = {'returncode': run.returncode, 'summary': summary.group(0) if summary else 'No test summary'}
    if run.returncode:
        raise RuntimeError('Tests failed; see tests.log')
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2))

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    names = [('fit_conditioned', 'One orbit per count'), ('fit_conditional_two_body', 'Orbit + density Jastrow'),
             ('target_two_orbits', 'Two orbits: direct'), ('witness_two_orbits', 'Two orbits: witness'),
             ('linear_rpa_q3', 'Static RPA q=3')]
    reference = next(r for r in data['pilot'] if r['G'] == -1.65 and r['kick'] == 'none')['rows']
    for ax, metric_name, ylabel in zip(axes, ('negativity', 'success_probability'), ('Filtered negativity', 'Filter success probability')):
        ax.plot([r['elapsed'] for r in reference], [r['filter_reference'][metric_name] for r in reference], 'k-', lw=2.4, label='Exact')
        for i, (family, label) in enumerate(names):
            rows = next(r for r in data['pilot'] if r['family'] == family and r['G'] == -1.65 and r['kick'] == 'none')['rows']
            ax.plot([r['elapsed'] for r in rows], [r['filter_prediction'][metric_name] for r in rows], 'o--', color=colors[i], ms=3, label=label)
        ax.set(xlabel='Time after freeze / interaction reversal', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7.5, loc='upper left')
    fig.suptitle('Representative pilot: L=12, G=2 to -1.65, no kick\nFilter selected from the earlier state; probabilities accompany conditional entanglement')
    save(fig, args.output, 'filtered_reversal')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    cases = [(G, kick) for G in (-1.25, -2.05) for kick in ('none', 'collective_sixth', 'collective_third')]
    x = np.arange(len(cases))
    for ax, metric_name, ylabel in zip(axes, ('negativity_error', 'unnormalized_filtered_trace_distance'), ('Absolute negativity error', 'Unnormalized filtered trace distance')):
        for i, (source, label) in enumerate((('target', 'Direct: two refined seeds'), ('witness', 'Witness: two refined seeds'))):
            values = np.array([[metric(primary(next(r for r in data['refinement'] if r['family'] == f'{source}_two_orbits_seed{seed}' and r['G'] == G and r['kick'] == kick)), metric_name)
                                for G, kick in cases] for seed in (20261111, 20261112)])
            ax.plot(x, values.mean(axis=0), 'o-', color=colors[i+2], label=label)
            ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=colors[i+2], alpha=.2)
        rpa = [metric(primary(next(r for r in data['confirmation'] if r['family'] == 'linear_rpa_q3' and r['G'] == G and r['kick'] == kick)), metric_name) for G, kick in cases]
        ax.plot(x, rpa, 's--', color=colors[4], label='Static RPA q=3')
        ax.set(xticks=x, xticklabels=[f'{G:g}\n'+{'none': '0', 'collective_sixth': 'pi/6', 'collective_third': 'pi/3'}[kick] for G, kick in cases], xlabel='New G / collective kick angle', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Prespecified reversal confirmation: elapsed time 1\nTraining-only convergence replication on the same six conditions; seeds are not new holdouts')
    save(fig, args.output, 'reversal_confirmation')

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    for i, L in enumerate((8, 12, 16)):
        rows = next(d for d in diagnostics if d['L'] == L and d['spectrum'] == 'picket')['rows']
        axes[0].plot([r['time'] for r in rows], [r['entanglement']['accessible_entropy_bits'] for r in rows], 'o-', color=colors[i], label=f'L={L}')
        axes[1].plot([r['time'] for r in rows], [r['filter']['negativity'] for r in rows], 'o-', color=colors[i])
        axes[2].semilogy([r['time'] for r in rows], [r['filter']['success_probability'] for r in rows], 'o-', color=colors[i])
    for ax, ylabel in zip(axes, ('Fixed-count accessible entropy (bits)', 'Band-edge filtered negativity', 'Filter success probability')):
        ax.set(xlabel='Original G=2 evolution time', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Offline size diagnostics: strong conditional entanglement can be rare\nTwo flat bands give zero fixed-count entropy on this grid')
    save(fig, args.output, 'interband_size_diagnostics')
    print(json.dumps({k: audit[k] for k in ('prediction_counts', 'frozen_files_verified', 'tests', 'max_norm_error', 'max_energy_drift', 'max_refinement_state_difference')}))
