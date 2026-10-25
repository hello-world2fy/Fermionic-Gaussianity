#!/usr/bin/env python3
"""Stage I screening, no Gaussian decomposition or target-trained dynamics."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import sys
import json
import hashlib
import subprocess
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.linear import unstable_modes, squeezed_pair_matrix, coupled_cluster_state, rank_truncate
from pairing_formation.structure import analyze, single_pbcs_count_distance_lower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--levels', nargs='+', type=int, default=[8, 12])
    parser.add_argument('--couplings', nargs='+', type=float, default=[1., 2., 4.])
    parser.add_argument('--points', type=int, default=121)
    parser.add_argument('--tau-max', type=float, default=6.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = Path(__file__).resolve().parents[1]
    provenance = {'command': sys.argv, 'git_head': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'python': sys.version, 'numpy': np.__version__,
        'source_sha256': {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(base.rglob('*.py'))}}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    summaries = []
    started = time.monotonic()
    for L in args.levels:
        for G in args.couplings:
            model = picket_fence(L, G)
            times = np.linspace(0., 2*args.tau_max/G, args.points)
            exact = model.evolve(times)
            linear_T = squeezed_pair_matrix(model, times)
            rows = []
            for i, (t, state, T) in enumerate(zip(times, exact, linear_T)):
                obs = model.observe(state)
                struct, radial, _ = analyze(model, state)
                full = coupled_cluster_state(model, T)
                oneT, singular = rank_truncate(T, 1)
                one = coupled_cluster_state(model, oneT)
                row = {'time': float(t), 'tau': float(G*t/2), **obs, **struct,
                       'linear_cc_fidelity': float(abs(np.vdot(state, full))**2),
                       'linear_rank1_cc_fidelity': float(abs(np.vdot(state, one))**2),
                       'linear_T_singular': singular.tolist(),
                       'linear_cc_observables': model.observe(full),
                       'radial_observables': model.observe(radial),
                       'pbcs_count_bound': single_pbcs_count_distance_lower(obs['count'])}
                rows.append(row)
            means = np.array([row['mean'] for row in rows])
            peaks = np.flatnonzero((means[1:-1] >= means[:-2]) & (means[1:-1] > means[2:]))+1
            peak = int(peaks[0]) if len(peaks) else int(np.argmax(means))
            max_excess = max(range(peak+1), key=lambda j: rows[j]['excess'])
            departure = next((j for j in range(peak+1) if rows[j]['linear_cc_fidelity'] < .99), peak)
            # Strong window: statistically forbidden onsite PBCS counts,
            # >=.99 single-channel arbitrary radial fidelity, mean >=.05.
            window = [j for j in range(peak+1) if rows[j]['excess'] > 1e-4
                      and rows[j]['single_direction_radial_fidelity'] >= .99
                      and rows[j]['mean'] >= .05]
            summary = {'L': L, 'G': G, 'dimension': model.dim,
                       'unstable_modes': unstable_modes(model),
                       'first_peak_index': peak, 'first_peak': rows[peak],
                       'max_excess_before_peak': rows[max_excess],
                       'first_linear_fidelity_below_99': rows[departure],
                       'window_time_interval': [rows[window[0]]['time'], rows[window[-1]]['time']] if window else None,
                       'max_norm_error': float(max(abs(np.linalg.norm(exact, axis=1)-1))),
                       'max_energy_drift': float(max(abs(np.einsum('ti,it->t', exact.conj(), model.H@exact.T)
                                                          -model.diagonal[model.fermi_index])))}
            name = f'L{L}_G{G:g}'
            (args.output/f'{name}.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2))
            np.savez_compressed(args.output/f'{name}_states.npz', times=times, states=exact,
                                linear_T=linear_T, energies=model.energies)
            summaries.append(summary)
            print(json.dumps({'case': name, 'window': summary['window_time_interval'],
                  'peak_t': rows[peak]['time'], 'peak_mean': rows[peak]['mean'],
                  'max_fano': max(x['fano'] for x in rows[:peak+1]),
                  'radial_F_peak': rows[peak]['single_direction_radial_fidelity'],
                  'rank1_upper_peak': rows[peak]['conditional_rank1_fidelity_upper'],
                  'seconds': time.monotonic()-started}), flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
