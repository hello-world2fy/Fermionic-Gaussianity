#!/usr/bin/env python3
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
from pairing_formation.interband import choose_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    saved = np.load(root/'replica2_L12_G2_01/t1.npz')
    target = saved['target']
    manifest = {'L': 12, 'train_time': 1., 'train_G': 2., 'holdout_G': [1.35, 2.65, -1.65],
                'kicks': {'none': 0., 'collective_quarter': float(np.pi/4), 'collective_half': float(np.pi/2)},
                'kick_kind': 'exp(-i alpha m), a one-body phase between the two bands',
                'elapsed': [0., .1, .25, .5, 1., 1.5, 2.], 'records': [],
                'primary_observables': ['filtered success probability', 'conditional negativity', 'unnormalized filtered density matrix trace distance', 'count TV', 'energy-resolved pair RMS'],
                'primary_time': 1., 'filters': {},
                'protocol': 'Filters and coordinates fixed from t=1 only; all nonlinear predictions saved before exact future references; canonical extraction is from the witness, not sampled atoms'}
    np.savez_compressed(args.output/'training.npz', state=target)
    for source in ('target', 'witness'):
        manifest['filters'][source] = choose_filter(model, saved[source])
        files = list((root/'multiorbit_fits_01').glob(f'{source}_t1_seed*.json'))
        chosen = max(files, key=lambda p: json.loads(p.read_text())['fidelity'])
        trained = np.load(chosen.with_suffix('.npz'))['coordinates']
        dynamics = MultiOrbitDynamics(model, 2)
        c, logs = dynamics.unpack(trained)
        initial, fidelity = dynamics.encode(logs, target)
        name = source+'_two_orbits'
        path = args.output/(name+'.npz')
        np.savez_compressed(path, coordinates=initial)
        manifest['records'].append({'name': name, 'file': path.name, 'type': 'multiorbit', 'rank': 2,
                                    'initial_fidelity': fidelity, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest['filters']['band_edges'] = {'particle': [0, model.r-1], 'hole': [0, model.r-1], 'selection': 'fixed energy extrema, independent of wavefunction'}
    for directory, selected in [
        ('jastrow_freeze_01', ['fit_conditioned', 'fit_two_body', 'fit_conditional_two_body', 'linear_rpa_q2', 'linear_rpa_q3', 'hamiltonian_R32']),
        ('conditional_dynamics_freeze_01', ['witness_conditioned'])]:
        old = json.loads((root/directory/'manifest.json').read_text())
        for record in old['records']:
            if record['name'] not in selected:
                continue
            source_path = root/directory/record['file']
            destination = args.output/source_path.name
            destination.write_bytes(source_path.read_bytes())
            kind = 'conditional' if record['type'] == 'tdvp' else record['type']
            manifest['records'].append({**record, 'type': kind, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'families': len(manifest['records']), 'predictions': 9*len(manifest['records']), 'filters': manifest['filters']}))
