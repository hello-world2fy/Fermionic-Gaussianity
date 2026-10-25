#!/usr/bin/env python3
"""Retrospective physical dark-sector diagnostic with existing strong controls."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.sparse import load_npz
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.dark_pairs import PairDarkSectors


def job(task):
    L, out = task
    root = out.parent
    model, forms = angular_model(L)
    dark = PairDarkSectors(model, forms)
    sources = {}
    states = {}
    times = None
    for method in ('exact', 'full_cc', 'source_only_cc', 'riccati'):
        path = root / 'nonlinear_channels_01' / f'L{L}_eta1_{method}.npz'
        saved = np.load(path)
        states[method] = saved['states']
        times = saved['times']
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if L in (14, 18):
        Q = load_npz(root / 'rotation_geometry_01' / f'L{L}_symmetry.npz')
        path = root / 'rotation_rank_controls_02' / f'L{L}_frozen_predictions.npz'
        saved = np.load(path)
        for name in ('matched', 'independent', 'Hamiltonian_block_response'):
            states[name] = (Q @ saved[name].T).T
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    results = {}
    for method, data in states.items():
        rows = []
        for t, raw in zip(times, data):
            state = raw / np.linalg.norm(raw)
            rows.append({'time': float(t), **dark.evaluate(state)})
        results[method] = rows
    result = {'L': L, 'G': 4., 'eta': 1., 'source_files': sources,
              'dark_sectors': dark.diagnostics, 'models': results}
    (out / f'L{L}_diagnostic.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / f'L{L}_band_projectors.npz',
                        **{f'm{m}': P for m, P in enumerate(dark.projectors)})
    print(json.dumps({'L': L, 'dark_dimensions': [x['dark_dimension'] for x in dark.diagnostics],
        'time2': {name: {key: rows[40][key] for key in ('total_both_dark', 'total_either_dark',
                     'multiple_transfer_both_dark', 'multiple_transfer_either_dark')}
                  for name, rows in results.items()}}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'protocol.json').write_text(json.dumps({
        'classification': 'Known-condition physical mechanism screen; no new physical forecasts or canonical-only claim.',
        'candidate': 'Excited band states annihilated by both original collective source pair annihilators.',
        'primary_time': 2., 'sizes': [10, 14, 18],
        'controls': ['full variational pair-CC', 'source-only CC', 'Riccati',
                     'matched Gaussian rotations', 'independent rotations', 'equal-rank Hamiltonian response'],
        'limitations': 'A band annihilator kernel is not a full-Hamiltonian invariant subspace. No claim of trapping, long lifetime, macroscopic subradiance or novel phase from population alone.'}, indent=2))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(job, (L, args.output)) for L in (10, 14, 18)]):
            f.result()
