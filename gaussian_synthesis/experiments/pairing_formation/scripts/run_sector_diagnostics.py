#!/usr/bin/env python3
"""Exact count/phase diagnostics, including an exactly collective control model.

This is a mechanism-screening experiment. It does not claim canonical-origin
discovery, holdout prediction superiority, or that counterflow is intrinsically
many-channel physics. Future states for phase interventions use H only.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import PairModel
from pairing_formation.sectors import (transfer_operator, resolved_observables,
    conditional_rank_one, phase_align, sector_components)


def job(task):
    L, G, spectrum, output = task
    x = 2*(np.arange(L)+.5)/L-1
    if spectrum == 'picket':
        energies = x
    elif spectrum == 'warped':
        energies = (x+.3*x**3)/1.3
    elif spectrum == 'asymmetric':
        energies = x+.18*(x*x-np.mean(x*x))
    elif spectrum == 'two_flat_bands':
        energies = .5*np.sign(x)
    else:
        raise ValueError(spectrum)
    model = PairModel(energies, G)
    operator = transfer_operator(model)
    times = np.linspace(0., 6*2/G, 121)
    states = model.evolve(times)
    mean = abs(states)**2@model.m
    maxima = np.flatnonzero((mean[1:-1] > mean[:-2]) & (mean[1:-1] >= mean[2:]))+1
    if not len(maxima):
        raise RuntimeError('No first pulse found')
    peak_index = int(maxima[0])
    def velocity(t):
        state = model.evolve([t])[0]
        return -2*model.g*model.r*np.vdot(state, operator@state).imag
    peak = brentq(velocity, times[peak_index-1], times[peak_index+1], xtol=2e-12)
    state = model.evolve([peak])[0]
    exact = resolved_observables(model, state, operator)
    rank_one = conditional_rank_one(model, state)
    aligned = phase_align(model, state, operator)
    controls = {'exact': state[:, None], 'time_reversal_mixture': np.column_stack((state, state.conj()))/np.sqrt(2),
                'sector_dephasing': sector_components(model, state),
                'phase_alignment': aligned[:, None], 'conditional_rank_one': rank_one[:, None]}
    elapsed = np.array([0., .1, .25, .5])*2/G
    curves = {}
    for name, components in controls.items():
        evolved = model.evolve(elapsed, components)
        rows = []
        for t, ensemble in zip(elapsed, evolved):
            obs = [resolved_observables(model, ensemble[:, j], operator) for j in range(ensemble.shape[1])]
            count = np.sum([q['count'] for q in obs], axis=0)
            m = np.arange(model.r+1)
            avg = count@m
            cross = sum(complex(sum(q['coherence_real']), sum(q['coherence_imag'])) for q in obs)
            rows.append({'elapsed': float(t), 'count': count.tolist(), 'mean': float(avg),
                         'variance': float(count@(m*m)-avg*avg),
                         'cross_real': float(cross.real), 'cross_imag': float(cross.imag),
                         'current': np.sum([q['current'] for q in obs], axis=0).tolist()})
        curves[name] = rows
    first = curves['exact'][0]
    count_error = max(np.max(abs(np.array(v[0]['count'])-first['count'])) for v in curves.values())
    twin = curves['time_reversal_mixture'][0]
    report = {'L': L, 'G': G, 'spectrum': spectrum, 'peak_time': peak, 'peak': exact,
              'conditional_rank_one_fidelity': float(abs(np.vdot(state, rank_one))**2),
              'initial_control_count_error': float(count_error),
              'time_reversal_mixture_initial_cross_error': float(abs(complex(twin['cross_real']-first['cross_real'],
                                                                                         twin['cross_imag']-first['cross_imag']))),
              'curves': curves,
              'interpretation': 'Generic finite quantum dynamics diagnostic; the flat-band control has exactly one collective direction per band'}
    name = f'L{L}_G{G:g}_{spectrum}'
    (output/(name+'.json')).write_text(json.dumps(report, indent=2))
    np.savez_compressed(output/(name+'.npz'), state=state, aligned=aligned, rank_one=rank_one,
                        energies=energies, peak_time=peak)
    print(json.dumps({k: report[k] for k in ('L', 'G', 'spectrum', 'peak_time', 'conditional_rank_one_fidelity')} |
                     {k: exact[k] for k in ('mean_rate', 'variance_rate', 'current_absolute_sum')}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=24)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(L, G, spectrum, args.output) for L in (8, 12, 16) for G in (2., 4.)
             for spectrum in ('picket', 'warped', 'asymmetric', 'two_flat_bands')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(job, tasks))
    (args.output/'summary.json').write_text(json.dumps([{k: v for k, v in r.items() if k != 'curves'} for r in results], indent=2))
