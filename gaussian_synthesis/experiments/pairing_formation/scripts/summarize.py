#!/usr/bin/env python3
"""Build figures and numerical audit from saved outputs, without rerunning fits."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'


def read(path):
    return json.loads((RESULTS/path).read_text())


def main():
    output = RESULTS/'assessment_01'
    output.mkdir(exist_ok=False)
    physics = read('physics_screen_01/L12_G2.json')
    rows = physics['rows']
    t = np.array([x['time'] for x in rows])
    select = t <= 3.
    fits = []
    for time in (.5, 1., 2.15):
        record = {'time': time}
        for family in ('pbcs', 'radial', 'radial_slope'):
            record[family] = read(f'key_fits_L12_G2_01/{family}_t{time:g}.json')['fit']['fidelity']
        for rank in (1, 2, 6):
            record[f'cc_rank{rank}'] = read(f'cc_L12_G2_01/t{time:g}_rank{rank}.json')['best_found_fidelity']
        record['general_projected_gaussian'] = max(read(f'general_gaussian_L12_G2_01/t{time:g}_seed{seed}.json')
                     ['best_found_fidelity'] for seed in (20261025, 20261026))
        fits.append(record)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for key, ax in zip(('mean', 'variance', 'cross_real', 'cross_imag'), axes.flat):
        ax.plot(t[select], [x[key] for x in rows if x['time'] <= 3], color='black', label='Exact')
        ax.plot(t[select], [x['linear_cc_observables'][key] for x in rows if x['time'] <= 3],
                '--', label='Linear boson / fermionic CC lift')
        for method in ('gaussian', 'discrete'):
            runs = [read(f'stochastic_L12_G2_01/{method}_seed{seed}_dt0.005.json') for seed in (20261025, 20261026)]
            tt = np.array([x['time'] for x in runs[0]['rows']])
            value = np.mean([[x[key] for x in run['rows']] for run in runs], axis=0)
            err = np.sqrt(np.sum([[x[key+'_se']**2 for x in run['rows']] for run in runs], axis=0))/2
            line, = ax.plot(tt, value, label='Gaussian-noise SMF' if method == 'gaussian' else 'Discrete TWA')
            ax.fill_between(tt, value-2*err, value+2*err, color=line.get_color(), alpha=.15)
        ax.axvline(1., color='gray', ls=':', lw=.8)
        ax.set(xlabel='Time', ylabel=key.replace('_', ' '))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Normal Fermi-sea quench: L=12 pair levels, G=2')
    fig.savefig(output/'physics_and_stochastic.png', dpi=180)
    fig.savefig(output/'physics_and_stochastic.pdf')
    plt.close(fig)

    evaluated = read('prediction_L12_G2_01/evaluation.json')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    colors = {1: '#e67e22', 2: '#2980b9', 3: '#27ae60'}
    for rec in evaluated:
        if rec['G'] != 2. or rec['source'] == 'target_m1':
            continue
        ls = '-' if rec['source'] == 'replica2_m1' else '--'
        source = 'replica witness' if ls == '-' else 'linear response'
        tt = [x['time'] for x in rec['rows']]
        for key, ax in zip(('fidelity', 'count_TV', 'cross_complex_error'), axes):
            ax.plot(tt, [x[key] for x in rec['rows']], ls, color=colors[rec['q']],
                    label=f"{source}, q={rec['q']}")
            ax.set(xlabel='Time', ylabel=key.replace('_', ' '))
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Frozen at t=1; no fitting of future states')
    fig.savefig(output/'frozen_predictions.png', dpi=180)
    fig.savefig(output/'frozen_predictions.pdf')
    plt.close(fig)

    physics_summaries = read('physics_screen_01/summary.json')+read('physics_L16_01/summary.json')
    max_norm = max(x['max_norm_error'] for x in physics_summaries)
    max_energy = max(x['max_energy_drift'] for x in physics_summaries)
    sampling = [read(f'full_haar_L12_t1_pilot_01/device{i}.json') for i in range(8)]
    reconstructions = [np.load(RESULTS/f'full_haar_L12_t1_pilot_01/device{i}_moments.npz')['reconstruction'] for i in range(8)]
    exact_C = np.load(RESULTS/'replica2_L12_G2_01/t1.npz')['reconstruction']
    timestep_max = {}
    for key in ('mean', 'variance', 'cross_real', 'cross_imag'):
        errors = []
        for method in ('gaussian', 'discrete'):
            for seed in (20261025, 20261026):
                coarse = read(f'stochastic_L12_G2_01/{method}_seed{seed}_dt0.01.json')['rows']
                fine = read(f'stochastic_L12_G2_01/{method}_seed{seed}_dt0.005.json')['rows']
                errors.extend(abs(a[key]-b[key]) for a, b in zip(coarse, fine))
        timestep_max[key] = max(errors)
    audit = {'fits': fits, 'max_exact_norm_error': max_norm, 'max_exact_energy_drift': max_energy,
             'timestep_halving_max_changes': timestep_max,
             'replica_residuals': {str(t): read(f'replica2_L12_G2_01/t{t:g}.json')['residual'] for t in (.5, 1., 2.15)},
             'full_haar_samples': sum(x['count'] for x in sampling),
             'canonical_B_pilot_mean': float(np.mean([x['B_mc'] for x in sampling])),
             'canonical_B_pilot_between_run_se': float(np.std([x['B_mc'] for x in sampling], ddof=1)/np.sqrt(8)),
             'combined_sampling_reconstruction_error': float(np.linalg.norm(np.mean(reconstructions, axis=0)-exact_C)),
             'phase1_physics_gate': True,
             'phase2_full_strong_baseline_gate': False,
             'phase3_unique_predictive_discovery_gate': False,
             'goal_complete': False}
    (output/'audit.json').write_text(json.dumps(audit, indent=2))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in str(p)
              and p.name != 'manifest_sha256.json'}
    (output/'manifest_sha256.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
