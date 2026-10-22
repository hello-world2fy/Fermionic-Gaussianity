"""Plot saved data only; can use a separate Python with matplotlib installed."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/anderson_four_mode_mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/anderson_four_mode_cache')
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'results/comparison_final'
FIGURES = ROOT/'figures'
FIGURES.mkdir(exist_ok=True)
GROUPS = {g['label']: g for g in json.loads((DATA/'comparison.json').read_text())}
BLUE, GREEN, ORANGE, RED, GRAY = '#2665A8', '#287C68', '#D18124', '#B74146', '#69717D'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 11, 'figure.dpi': 150, 'savefig.dpi': 180})


def trajectory(label):
    index = GROUPS[label]['case_indices'][0]
    with (DATA/f'case_{index:03d}.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(r[key]) if r[key] else np.nan for r in rows]) for key in rows[0]}


def save(fig, name):
    fig.savefig(FIGURES/f'{name}.png', bbox_inches='tight')
    fig.savefig(FIGURES/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.7), constrained_layout=True)
    ax = axes[0, 0]
    for key, label, color, style in [
        ('strang', 'Exact reference', GRAY, '-'),
        ('newton:two_node', 'Original Newton + 2 nodes', GREEN, '--'),
        ('api:sample_refit', 'Original API + sampled refit', BLUE, ':'),
        ('newton:sample_raw', 'Raw measure sampling (fails)', RED, '-.')]:
        r = trajectory(key)
        ax.plot(r['time'], r['exact_double'] if key == 'strang' else r['double'],
                label=label, color=color, ls=style, lw=1.8)
    ax.set(xlabel='Time', ylabel=r'$D(t)$', title='A. Actual closed-loop evolution; seed 20261023')
    ax.legend(fontsize=8, loc='lower right')

    ax = axes[0, 1]
    for key, label, color, style in [
        ('strang', 'Strang error floor', GRAY, '-'),
        ('newton:two_node', 'Original Newton + 2 nodes', GREEN, '--'),
        ('sgd:two_node', 'Original SGD + 2 nodes (fails)', RED, '-'),
        ('sgd_polished:two_node', 'Original SGD + Newton polish', BLUE, ':')]:
        r = trajectory(key)
        ax.semilogy(r['time'][1:], np.maximum(r['infidelity'][1:], 1e-15),
                    label=label, color=color, ls=style, lw=1.8)
    ax.axhline(1e-5, color=ORANGE, ls=':', lw=1, label='Infidelity acceptance threshold')
    ax.set(xlabel='Time', ylabel=r'$1-|\langle\psi_{\rm exact}|\hat\psi\rangle|^2$',
           title='B. Solver residual must be corrected before compression')
    ax.legend(fontsize=8, loc='lower right')

    ax = axes[1, 0]
    keys = ['strang', 'schmidt', 'branch_omp', 'direct_quadric', 'haar:sample_refit',
            'newton:two_node', 'newton:sample_refit', 'api:sample_refit']
    labels = ['Dense Strang', 'Schmidt / SVD', 'Branch OMP', 'Direct quadric (no solve)',
              'Haar + refit', 'Original Newton + 2 nodes', 'Original Newton + sampling',
              'Original API + sampling']
    colors = [GRAY]*5+[GREEN, BLUE, BLUE]
    for i, key in enumerate(keys):
        g = GROUPS[key]
        value = g['median_total_seconds']*1000
        lo, hi = [v*1000 for v in g['time_range']]
        ax.barh(i, value, color=colors[i], alpha=.88)
        ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], color='black', capsize=2, lw=.8)
        ax.text(max(value, hi)*1.2, i, f'{value:.3g}', va='center', fontsize=9)
    ax.set_yticks(range(len(keys)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set(xscale='log', xlim=(.1, 5000), xlabel='Total elapsed time (ms; median, range of 3 runs)',
           title='C. Same accuracy: all shown methods pass both thresholds')

    ax = axes[1, 1]
    keys = ['haar:sample_refit', 'unoptimized:sample_refit', 'api:sample_refit',
            'newton:sample_refit', 'newton:two_node', 'schmidt']
    labels = ['Haar', 'Overlap bias\n(no solve)', 'API\n+ samples',
              'Newton\n+ samples', 'Newton\n+ 2 nodes', 'Schmidt']
    for i, key in enumerate(keys):
        values = GROUPS[key]['mean_postcompression_cost']
        ax.bar(i, np.mean(values), color=GREEN if i >= 4 else BLUE, alpha=.78)
        ax.scatter(np.array([-.09, 0, .09])+i, values, color='black', s=12, zorder=3)
    ax.set_xticks(range(len(keys)), labels, fontsize=8)
    ax.set(ylabel=r'Mean post-compression $B^2=(\sum |c_j|)^2/\|\psi\|^2$',
           ylim=(0, 6.4), title='D. Overlap bias explains most sampling benefit')
    fig.suptitle('Four fermionic modes: the original decomposition is in the propagation loop\n'
                 r'$U=4,\ V=0.6,\ T=2,\ \Delta t=0.05$; compression every 4 steps', fontsize=14)
    save(fig, 'overview')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    ax = axes[0]
    keys = ['api:sample_refit', 'newton:sample_refit', 'newton:two_node']
    components = [('solve', 'Solve + original validation', BLUE),
                  ('sampling', 'Proposals + mass estimate', ORANGE),
                  ('refit', 'Coefficient refit', GREEN),
                  ('moment_compression', 'Two-node construction', '#8A60A2')]
    for i, key in enumerate(keys):
        # Use one actual median-time run so components add to its total.
        cases = [json.loads((DATA/f'case_{j:03d}.json').read_text())['summary']
                 for j in GROUPS[key]['case_indices']]
        case = sorted(cases, key=lambda c: c['total_seconds'])[1]
        bottom = 0.
        for field, label, color in components:
            height = case['timing_components'][field]/case['total_seconds']*100
            ax.bar(i, height, bottom=bottom, color=color, label=label if i == 0 else None)
            bottom += height
        ax.bar(i, 100-bottom, bottom=bottom, color=GRAY, label='Setup + propagation + other' if i == 0 else None)
        ax.text(i, 102, f"{case['total_seconds']*1000:.3g} ms", ha='center', fontsize=9)
    ax.set_xticks(range(3), ['API +\nsampling', 'Newton +\nsampling', 'Newton +\n2 nodes'])
    ax.set(ylabel='Share of complete elapsed time (%)', ylim=(0, 111), title='A. All solve and sampling costs are charged')
    ax.legend(fontsize=7, loc='lower left')

    controls = json.loads((ROOT/'results/controls/summary.json').read_text())
    spacings = [controls[j] for j in (13, 14, 15, 16)]
    main_case = json.loads((DATA/f"case_{GROUPS['newton:two_node']['case_indices'][0]:03d}.json").read_text())['summary']
    spacings = sorted(spacings+[main_case], key=lambda c: c['config']['block_steps'])
    x = [c['config']['block_steps'] for c in spacings]
    ax = axes[1]
    ax.plot(x, [c['total_seconds']*1000 for c in spacings], 'o-', color=BLUE)
    ax.set(xlabel='Steps between compressions', ylabel='Total elapsed time (ms)',
           title='B. Less frequent solves reduce total time', xticks=x)
    second = ax.twinx()
    second.semilogy(x, [c['peak_branch_atoms'] for c in spacings], 's--', color=ORANGE)
    second.set_ylabel('Peak propagated branches', color=ORANGE)
    ax.text(.05, .90, 'One seed; exploratory spacing sweep', transform=ax.transAxes, fontsize=8)

    ax = axes[2]
    for key, label, color in [('api:sample_refit', 'API + sampled refit', BLUE),
                              ('newton:two_node', 'Newton + 2 nodes', GREEN),
                              ('newton:sample_raw', 'Raw measure sampling (fails)', RED)]:
        r = trajectory(key)
        ax.step(r['time'], r['atoms'], where='post', label=label, color=color, lw=1.3)
    ax.set(xlabel='Time', ylabel='Branches after each step', yscale='log',
           title='C. Branch growth between compression events')
    ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('Runtime includes setup, solver, validation, proposals, mass estimate, refit and observables', fontsize=12)
    save(fig, 'cost_breakdown')
    print(FIGURES/'overview.png')
    print(FIGURES/'cost_breakdown.png')


if __name__ == '__main__':
    main()
