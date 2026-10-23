#!/usr/bin/env python3
"""Plot saved pilot data only; no fitting, sampling, or time evolution."""
import _bootstrap
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    output = args.directory
    # Keep Matplotlib caches out of the user's home and out of experiment artifacts.
    with tempfile.TemporaryDirectory(prefix='bcs-matplotlib-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from gaussian_synthesis import fidelity
        from bcs_pairing.provenance import digest, write_json

        with np.load(output/'states.npz', allow_pickle=False) as saved:
            a = {key: saved[key] for key in saved.files}
        with (output/'baselines.csv').open() as stream:
            quench = list(csv.DictReader(stream))
        with (output/'asymptotic.csv').open() as stream:
            asym = list(csv.DictReader(stream))
        config = json.loads((output/'config.json').read_text())
        figure_dir = output/'figures'
        figure_dir.mkdir(exist_ok=False)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        ax = axes[0, 0]
        for name, label in [('td_bcs', 'Projected TD-BCS'), ('product_tdvp', 'Projected product TDVP')]:
            ax.plot(a['times'], [fidelity(x, y) for x, y in zip(a['exact'], a[name])], label=label)
        t = np.array([float(row['time']) for row in quench])
        found = np.array([float(row['F_PBCS_found']) for row in quench])
        upper = np.array([float(row['F_PBCS_upper']) for row in quench])
        ax.scatter(t, found, color='black', marker='o', s=25, label='Refitted onsite PBCS')
        ax.set(xlabel='Time (1/W)', ylabel='Fidelity', title='Quench: trajectory versus refitted ansatz')
        ax.legend(fontsize=8, loc='lower left')
        ax = axes[0, 1]
        ax.semilogy(t[1:], 1-found[1:], 'o-', label='Achieved infidelity')
        ax.semilogy(t[1:], 1-upper[1:], 's--', label='Lower bound on best onsite infidelity')
        ax.fill_between(t[1:], 1-upper[1:], 1-found[1:], alpha=.12)
        ax.set(xlabel='Time (1/W)', ylabel='Infidelity', title='Onsite PBCS: achievable / rigorous bounds')
        ax.legend(fontsize=8)
        ax = axes[1, 0]
        rows = [row for row in asym if float(row['eta']) == config['alpha_scan_eta']]
        alpha = np.array([float(row['alpha']) for row in rows])
        lower_error = np.array([1-float(row['F_PBCS_upper']) for row in rows])
        found_error = np.array([1-float(row['F_PBCS_found']) for row in rows])
        ax.plot(alpha, found_error, 'o-', label='Achieved infidelity')
        ax.plot(alpha, lower_error, 's--', label='Lower bound on best onsite infidelity')
        ax.fill_between(alpha, lower_error, found_error, alpha=.12)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set(xlabel='Jastrow phase multiplier alpha', ylabel='Infidelity', title='Asymptotic state: eta=0.8, fixed probabilities')
        ax.legend(fontsize=8)
        ax = axes[1, 1]
        for t0, states in zip(a['inverse_t0'], a['inverse_states']):
            error = [1-fidelity(x, y) for x, y in zip(states, a['inverse_asymptotes'])]
            ax.loglog(a['inverse_times'], error, 'o-', label=f't0={t0:g}')
        ax.set(xlabel='Final time (1/W)', ylabel='Infidelity to asymptotic state', title='1/t coupling: direct dynamics cross-check')
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.suptitle('BCS baseline pilot | L=4, m=2, W=1 | quench G=0.6 (lambda=2.4)', fontsize=13)
        fig.savefig(figure_dir/'overview.png', dpi=170)
        fig.savefig(figure_dir/'overview.svg')
        plt.close(fig)
        script_copy = figure_dir/'plot_results.py'
        script_copy.write_bytes(Path(__file__).read_bytes())
        write_json(figure_dir/'manifest.json', {'matplotlib': matplotlib.__version__,
                    'inputs': {name: digest(output/name) for name in ('config.json', 'states.npz', 'baselines.csv', 'asymptotic.csv')},
                    'outputs': {name: digest(figure_dir/name) for name in ('overview.png', 'overview.svg', 'plot_results.py')}})
        print(figure_dir/'overview.png')


if __name__ == '__main__':
    main()
