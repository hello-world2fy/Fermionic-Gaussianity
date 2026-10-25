#!/usr/bin/env python3
"""Prospective follow-up to the interaction-reversal pilot, with both fit seeds."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.multiorbit import MultiOrbitDynamics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fits-directory', default='multiorbit_fits_01')
    parser.add_argument('--refinement', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    prior = root/'interband_freeze_01'
    previous = json.loads((prior/'manifest.json').read_text())
    model = picket_fence(12, 2.)
    target = np.load(prior/'training.npz')['state']
    np.savez_compressed(args.output/'training.npz', state=target)
    manifest = {'L': 12, 'train_time': 1., 'train_G': 2., 'holdout_G': [-1.25, -2.05],
                'kicks': {'none': 0., 'collective_sixth': float(np.pi/6), 'collective_third': float(np.pi/3)},
                'elapsed': [0., .1, .25, .5, 1., 1.5, 2.], 'records': [], 'filters': previous['filters'],
                'primary_time': 1., 'primary_metrics': ['absolute conditional negativity error', 'unnormalized filtered density trace distance'],
                'secondary_metrics': ['count TV', 'energy-resolved pair RMS', 'fidelity'],
                'hypothesis': 'The modest canonical-witness advantage for filtered entanglement in the G=-1.65 pilot persists at two fresh interaction reversals and across both training-fit seeds',
                'failure_rule': 'No stable advantage if the two primary metrics disagree systematically or one fit seed reverses the ranking; do not substitute a different observable',
                'protocol': 'Follow-up motivated by an inspected pilot, but these two negative couplings and kick combinations have not been evaluated; all future predictions precede exact references'}
    if args.refinement:
        manifest['protocol'] = 'Training-only convergence refinement on the already frozen confirmation grid; this is a numerical replication, not an additional independent holdout. Both sources receive the same larger optimization budget.'
    for source in ('target', 'witness'):
        for seed in (20261111, 20261112):
            fitted = np.load(root/args.fits_directory/f'{source}_t1_seed{seed}.npz')['coordinates']
            dynamics = MultiOrbitDynamics(model, 2)
            c, logs = dynamics.unpack(fitted)
            coordinates, fidelity = dynamics.encode(logs, target)
            name = f'{source}_two_orbits_seed{seed}'
            path = args.output/(name+'.npz')
            np.savez_compressed(path, coordinates=coordinates)
            manifest['records'].append({'name': name, 'file': path.name, 'type': 'multiorbit', 'rank': 2,
                                        'source_family': source, 'fit_seed': seed, 'initial_fidelity': fidelity,
                                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for record in previous['records']:
        if args.refinement:
            continue
        if record['name'] not in ('fit_conditioned', 'fit_conditional_two_body', 'linear_rpa_q3'):
            continue
        path = args.output/record['file']; path.write_bytes((prior/record['file']).read_bytes())
        manifest['records'].append({**record, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'families': len(manifest['records']), 'new_predictions': 6*len(manifest['records']), 'primary_time': manifest['primary_time']}))
