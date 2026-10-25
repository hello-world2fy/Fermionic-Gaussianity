#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.multiorbit import MultiOrbitDynamics, fit_orbits
from pairing_formation.replica2 import Replica2
from pairing_formation.interband import choose_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=int, choices=[8, 16], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(args.levels, 2.)
    target = np.load(root/f'conditional_scale_L{model.L}_freeze_01/training.npz')['state']
    np.savez_compressed(args.output/'training.npz', state=target)
    sources = {'target': target}
    if model.L == 8:
        witness, report = Replica2(model).solve(target)
        sources['witness'] = witness
        (args.output/'replica_report.json').write_text(json.dumps(report, indent=2))
        np.savez_compressed(args.output/'replica.npz', target=target, witness=witness)
    manifest = {'L': model.L, 'train_time': 1., 'train_G': 2., 'holdout_G': [1.35, 2.65, -1.65],
                'kicks': {'none': 0.}, 'elapsed': [0., .1, .25, .5, 1., 1.5, 2.], 'records': [],
                'primary_time': 1., 'protocol': 'Size replication of training-only orbital extraction and fixed energy filters; L16 has no canonical-witness comparison',
                'filters': {name: choose_filter(model, state) for name, state in sources.items()}}
    for source, state in sources.items():
        report, fitted = fit_orbits(model, state, rank=2, starts=6, seed=20261113)
        dynamics = MultiOrbitDynamics(model, 2)
        c, logs = dynamics.unpack(fitted)
        initial, fidelity = dynamics.encode(logs, target)
        name = source+'_two_orbits'
        path = args.output/(name+'.npz')
        np.savez_compressed(path, coordinates=initial)
        (args.output/(name+'_fit.json')).write_text(json.dumps(report, indent=2))
        manifest['records'].append({'name': name, 'file': path.name, 'type': 'multiorbit', 'rank': 2,
                                    'initial_fidelity': fidelity, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    previous = root/f'conditional_scale_L{model.L}_freeze_01/target_conditioned.npz'
    path = args.output/'fit_conditioned.npz'
    path.write_bytes(previous.read_bytes())
    manifest['records'].append({'name': 'fit_conditioned', 'file': path.name, 'type': 'conditional', 'conditioned': True,
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'L': model.L, 'families': len(manifest['records']), 'fit_fidelities': [r.get('initial_fidelity') for r in manifest['records']]}))
