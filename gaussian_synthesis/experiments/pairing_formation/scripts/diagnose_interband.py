#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence, PairModel
from pairing_formation.interband import sector_entanglement, EntanglingSources, filtered_density, tomography, schmidt_dephasing_columns


def job(task):
    L, flat, output = task
    model = PairModel(np.r_[-.5*np.ones(L//2), .5*np.ones(L//2)], 2.) if flat else picket_fence(L, 2.)
    analyzer = EntanglingSources(model)
    times = np.array([.5, 1., 1.5, 2., 2.15, 2.5, 3., 4.])
    states = model.evolve(times)
    rows = []
    for t, state in zip(times, states):
        source, product = analyzer.analyze(state)
        rho, probability = filtered_density(model, state, [0, model.r-1], [0, model.r-1])
        # These are offline diagnostics, never training data for held-out dynamics.
        phases = []
        for alpha in (0., np.pi/4, np.pi/2):
            shifted, _ = analyzer.analyze(product*np.exp(-1j*alpha*model.m))
            phases.append({'alpha': float(alpha), 'normal_squared_norm': shifted['total_normal_squared_norm']})
        rows.append({'time': float(t), 'counting': model.observe(state), 'entanglement': sector_entanglement(model, state),
                     'filter': tomography(rho, probability), 'sources': source, 'phase_control_of_rank1_source': phases})
    name = f'L{L}_'+('flat' if flat else 'picket')
    (output/(name+'.json')).write_text(json.dumps({'L': L, 'spectrum': 'flat' if flat else 'picket', 'rows': rows}, indent=2))
    return {'name': name, 'max_accessible_entropy': max(x['entanglement']['accessible_entropy_bits'] for x in rows),
            'max_local_source_leakage': max(y['local_normal_norm'] for x in rows for y in x['sources']['records'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(L, flat, args.output) for L in (8, 12, 16) for flat in (False, True)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(job, task) for task in tasks]):
            print(json.dumps(future.result()), flush=True)
    # A physical local CPTP deletion, with explicit limitations: state-adapted
    # Schmidt dephasing also changes some coherence between transfer sectors.
    model = picket_fence(12, 2.)
    state = model.evolve([2.15])[0]
    columns = schmidt_dephasing_columns(model, state)
    times = [0., .1, .25, .5]
    future_pure = model.evolve(times, state)
    future_columns = model.evolve(times, columns)
    control = []
    for t, pure, mixed in zip(times, future_pure, future_columns):
        rp, pp = filtered_density(model, pure, [0, 5], [0, 5])
        rm, pm = filtered_density(model, mixed, [0, 5], [0, 5])
        pm_count = np.bincount(model.m, weights=np.sum(abs(mixed)**2, axis=1), minlength=model.r+1)
        pure_count = np.array(model.observe(pure)['count'])
        control.append({'elapsed': t, 'pure_filter': tomography(rp, pp), 'dephased_filter': tomography(rm, pm),
                        'count_TV': float(np.sum(abs(pm_count-pure_count))/2)})
    (args.output/'schmidt_dephasing.json').write_text(json.dumps({'protocol': 'Local CPTP, count- and local-reduced-state-preserving deletion at t=2.15; state-adapted basis; also changes some inter-m coherences', 'rows': control}, indent=2))
