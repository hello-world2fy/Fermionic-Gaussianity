#!/usr/bin/env python3
"""Freeze training-only families and a genuinely new parameter/kick grid."""
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
from pairing_formation.fitting import excitation_features
from pairing_formation.conditional_dynamics import ConditionalDynamics
from pairing_formation.jastrow import JastrowDynamics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    target = np.load(root/'replica2_L12_G2_01/t1.npz')['target']
    common_saved = np.load(root/'conditional_dynamics_freeze_01/target_common.npz')['coordinates']
    common = ConditionalDynamics(model, False)
    common_state = common.state_and_blocks(common_saved)[0]
    c, z = common.unpack(common_saved)
    raw = excitation_features(model)@z[0]
    manifest = {'L': 12, 'train_time': 1., 'train_G': 2., 'holdout_G': [1.45, 2.55, -1.7],
                'kicks': {'none': 0., 'spectral_plus': .35, 'spectral_minus': -.35},
                'elapsed': [0., .25, .5, 1., 1.5, 2.], 'records': [],
                'protocol': 'Freeze all parameter families and training-only fits; propagate by Hamiltonian TDVP; save every prediction before accessing any exact future reference. Shared families all start at the same approximate physical state.',
                'primary_comparison': 'positive-G elapsed=1 count TV and energy-resolved pair RMS; elapsed=2 and negative G are stress tests',
                'no_equal_runtime_advantage_claim': True}
    np.savez_compressed(args.output/'training.npz', state=target)

    def save(name, coordinates, dynamics, kind, shared=False):
        path = args.output/(name+'.npz')
        if kind == 'jastrow':
            np.savez_compressed(path, coordinates=coordinates, features=dynamics.features)
            extra = {'jastrow_family': dynamics.family, 'complex_cone_dimension': len(coordinates)}
        else:
            np.savez_compressed(path, coordinates=coordinates)
            extra = {'conditioned': True, 'complex_cone_dimension': 57}
        state = dynamics.state_and_blocks(coordinates)[0]
        record = {'name': name, 'file': path.name, 'type': kind, **extra,
                  'shared_initial_state': shared,
                  'initial_fidelity': float(abs(np.vdot(target, state))**2),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if shared:
            record['shared_initial_state_difference'] = float(np.linalg.norm(state-common_state))
            assert record['shared_initial_state_difference'] < 1e-11
        manifest['records'].append(record)

    for family in ('collective_0', 'collective_1', 'collective_2', 'two_body', 'collective_3', 'conditional_two_body'):
        folder = root/('jastrow_fits_extended_01' if family in ('collective_3', 'conditional_two_body') else 'jastrow_fits_01')
        # Choose only the better training-time fit. Pulse fits never initialize predictions.
        candidates = list(folder.glob(f'{family}_t1_seed*.json'))
        chosen = max(candidates, key=lambda p: json.loads(p.read_text())['fidelity'])
        saved = np.load(chosen.with_suffix('.npz'))
        dynamics = JastrowDynamics(model, family, features=saved['features'])
        save('fit_'+family, saved['coordinates'], dynamics, 'jastrow')
        theta, error = dynamics.project_logarithms(raw)
        assert error < 1e-11
        coordinates, _ = dynamics.encode(theta, common_state)
        save('shared_'+family, coordinates, dynamics, 'jastrow', True)
    dynamics = ConditionalDynamics(model, True)
    coordinates = np.load(root/'conditional_dynamics_freeze_01/target_conditioned.npz')['coordinates']
    save('fit_conditioned', coordinates, dynamics, 'conditional')
    save('shared_conditioned', np.r_[c, np.repeat(z, model.r+1, axis=0).ravel()], dynamics, 'conditional', True)
    previous = json.loads((root/'conditional_dynamics_freeze_01/manifest.json').read_text())
    for record in previous['records']:
        if record['type'] == 'static':
            source = root/'conditional_dynamics_freeze_01'/record['file']
            destination = args.output/source.name
            destination.write_bytes(source.read_bytes())
            manifest['records'].append({**record, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'families': len(manifest['records']), 'predictions': 9*len(manifest['records']),
                      'max_shared_initial_state_difference': max(r.get('shared_initial_state_difference', 0.) for r in manifest['records'])}))
