#!/usr/bin/env python3
"""Audit the saved round-sixteen evidence and produce physical comparisons."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = out.parent
    experiment = root.parent
    load = lambda path: json.loads(path.read_text())
    verified = {}

    def verify(path, expected):
        path = Path(path)
        if not path.is_absolute():
            path = experiment / path
        assert digest(path) == expected, str(path)
        verified[str(path.relative_to(experiment))] = expected

    all_rows, all_gates, predictions, protocols = [], [], [], []
    for directory in ('pair_symbol_pilot_01', 'pair_contrast_01'):
        source = root / directory
        protocol = load(source / 'protocol.json')
        frozen = load(source / 'frozen_predictions.json')
        evaluation = load(source / 'evaluation.json')
        protocols.append(protocol)
        for path, value in protocol['input_hashes'].items():
            verify(path, value)
        verify(source / 'frozen_spaces.npz', protocol['frozen_sha256'])
        if 'training_sha256' in protocol:
            verify(source / 'training.npz', protocol['training_sha256'])
        if 'baseline_evaluation' in protocol:
            verify(protocol['baseline_evaluation'], protocol['baseline_evaluation_sha256'])
        for row in frozen:
            verify(source / row['file'], row['sha256'])
        assert [(r['file'], r['sha256']) for r in frozen] == [
            (r['file'], r['sha256']) for r in evaluation['models']]
        all_rows.extend(evaluation['models'])
        all_gates.extend(evaluation['gate'])
        predictions.extend(frozen)

    jobs = load(root / 'pair_symbol_pilot_01/moment_jobs.json')
    for row in jobs:
        verify(root / 'pair_symbol_pilot_01' / row['file'], row['sha256'])
    haar = []
    haar_batches = []
    for directory in ('pair_symbol_checks_01', 'pair_symbol_checks_02'):
        rows = load(root / directory / 'summary.json')['jobs']
        for row in rows:
            verify(root / directory / row['file'], row['sha256'])
        haar.extend(rows)
        haar_batches.append({'directory': directory,
            'samples': sum(r['samples'] for r in rows),
            'max_error_over_estimated_Frobenius_SE': max(
                c['error_over_SE'] for r in rows for c in r['checks'])})
    diagnostics = load(root / 'pair_symbol_diagnostics_01/diagnostics.json')
    for path, value in diagnostics['input_hashes'].items():
        verify(path, value)
    log_path = root / 'pair_symbol_tests_01.log'
    log = log_path.read_text()
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log)
    assert match
    verify(log_path, digest(log_path))
    shutil.copyfile(log_path, out / 'tests.log')

    geometry = load(root / 'pair_symbol_pilot_01/geometry.json')
    audit = {
        'classification': 'Sixteenth round: progress; important canonical-specific physics objective remains active and unmet.',
        'actual_full_Fock_Gaussian_pair_symbols': True,
        'physical_modes': 28, 'GPUs_used': 8,
        'Haar_check_samples': sum(r['samples'] for r in haar),
        'Haar_check_batches': haar_batches, 'iid_canonical_measure_draws': 0,
        'training_times': protocols[0]['training_times'],
        'reused_canonical_syntheses': 4, 'fresh_canonical_syntheses': 0,
        'pair_feature_cases': sum(r['pairs'] for r in jobs),
        'coherent_component_vectors': 3 * sum(r['pairs'] for r in jobs),
        'full_paired_dimension': 3432, 'invariant_dimension': 170,
        'shared_dimension': protocols[0]['base_dimension'],
        'primary_pilot_trajectories': len(predictions) - 8,
        'posthoc_contrast_trajectories': 8,
        'total_saved_trajectories': len(predictions),
        'canonical_gates': len(all_gates),
        'passing_canonical_gates': [r for r in all_gates if r['passes']],
        'independent_exponential_repeats': len(diagnostics['independent_exponential_repeats']),
        'same_initial_deletion_starts': ['Fermi', 'restart_at_1.35'],
        'new_physical_parameter_holdouts': 0,
        'new_important_physics_established': False,
        'canonical_unique_predictive_structure_established': False,
        'novelty_priority_established': False,
        'equal_total_compute_advantage_established': False,
        'rigorous_MC_confidence_certificate': False,
        'interval_certified_physical_projection_bound': False,
        'P3_P4_scope': 'Independent small-Fock normalization checks only; the synthesis remains k2.',
        'contrast_scope': 'Offsite coherent M_anomalous-M_full/3 vanishes on paired Slater witnesses; its label is not pointwise zero and is not a target observable marginal.',
        'max_reused_canonical_synthesis_residual': max(r['synthesis_residual'] for r in jobs if r['method'] == 'canonical'),
        'max_component_split_residual': max(r['split_sum_residual'] for r in jobs),
        'max_initial_state_error': max(r['initial_error'] for r in predictions),
        'max_norm_error': max(r['norm_error'] for r in predictions),
        'max_independent_exponential_state_difference': max(r['max_state_difference'] for r in diagnostics['independent_exponential_repeats']),
        'max_nested_deletion_residual': max(r['nested_anomalous_residual'] for r in diagnostics['nested_deletion_geometry']),
        'tests_passed': int(match[1]), 'tests_seconds': float(match[2]),
        'verified_file_count': len(verified), 'verified_files': verified,
        'primary': [{'method': r['method'], 'start': r['start'], 'dimension': r['dimension'], **r['primary']} for r in all_rows],
        'capacity': diagnostics['capacity'],
        'nested_deletion_geometry': diagnostics['nested_deletion_geometry'],
        'canonical_unoptimized_cosines': geometry['canonical_vs_unoptimized'],
        'remaining_gate': 'Identify important coherent physics missed by strong conventional methods, explain it through Gaussian configurations, then pass genuinely unseen conditions and same-initial-state deletion.'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))

    lookup = {(r['method'], r['start']): r for r in all_rows}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    styles = [
        ('canonical_anomalous_q8', 'Anomalous, d=17', '#0072B2', '-'),
        ('canonical_split_q16', 'Anomalous + normal, d=25', '#D55E00', '-'),
        ('canonical_contrast_q16', 'Contrast, d=25', '#CC79A7', '-'),
        ('unoptimized_contrast_q16', 'Unoptimized contrast, d=25', '#009E73', ':'),
        ('canonical_anomalous_H_q16', 'Anomalous + H, d=25', '#E69F00', '--'),
        ('Hamiltonian_q16', 'Hamiltonian, d=25', 'black', '-.')]

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{ext}', dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for method, label, color, line in styles:
        rows = lookup[method, 'restart']['rows']
        times = [r['time'] for r in rows]
        for ax, key in zip(axes, ('count_TV', 'pair_channel_L2_error', 'multiple_dark_probability')):
            values = [r[key] for r in rows]
            if key != 'multiple_dark_probability':
                values = np.maximum(values, 1e-14)
            ax.plot(times, values, line, label=label, color=color, lw=1.5)
    rows = lookup['exact_symmetry', 'restart']['rows']
    axes[2].plot([r['time'] for r in rows], [r['multiple_dark_probability'] for r in rows], color='0.6', lw=3, alpha=.7, label='Exact')
    for ax, title in zip(axes, ('Count total-variation error', 'Pair-channel L2 error', 'Multiple-transfer dark probability')):
        ax.set(title=title, xlabel='Time', xlim=(1.4, 2.5))
        ax.axvline(2, color='0.75', ls=':', lw=.8)
    for ax in axes[:2]:
        ax.set(yscale='log', ylim=(1e-12, 2))
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .82))
    save(fig, 'pair_symbol_predictions')

    selected = ['canonical_anomalous_q8', 'canonical_split_q16', 'canonical_contrast_q16', 'unoptimized_contrast_q16', 'canonical_anomalous_H_q16', 'Hamiltonian_q16']
    labels = ['Anomalous\nd=17', 'Split\nd=25', 'Contrast\nd=25', 'Unoptimized\ncontrast d=25', 'Anomalous + H\nd=25', 'Hamiltonian\nd=25']
    capacity = {(r['method'], r['time']): r for r in diagnostics['capacity']}
    x = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, (key, label) in enumerate((('count_TV', 'Count TV'), ('pair_channel_L2_error', 'Pair-channel L2'), ('multiple_dark_error', 'Absolute dark error'))):
        axes[0].bar(x + (j - 1) * .25, [abs(lookup[m, 'restart']['primary'][key]) for m in selected], .25, label=label)
    axes[0].set(yscale='log', ylabel='Error at t=2, identical restart', ylim=(1e-11, 1))
    axes[0].legend(fontsize=8)
    axes[1].bar(x - .18, [capacity[m, 2.]['propagation_fidelity'] for m in selected], .36, label='Frozen-space propagation')
    axes[1].bar(x + .18, [capacity[m, 2.]['subspace_max_fidelity'] for m in selected], .36, label='Best overlap in saved space')
    axes[1].set(ylabel='Fidelity at t=2', ylim=(0, 1.05))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set(xticks=x, xticklabels=labels)
        ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    save(fig, 'pair_symbol_deletion_capacity')
    print(json.dumps({k: audit[k] for k in ('Haar_check_samples', 'total_saved_trajectories', 'canonical_gates', 'passing_canonical_gates', 'max_initial_state_error', 'max_norm_error', 'max_reused_canonical_synthesis_residual', 'max_component_split_residual', 'tests_passed', 'verified_file_count')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
