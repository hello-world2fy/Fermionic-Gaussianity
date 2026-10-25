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
from pairing_formation.fitting import excitation_features
from pairing_formation.conditional_dynamics import ConditionalDynamics
from pairing_formation.jastrow import JastrowDynamics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=int, required=True, choices=[8, 16])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    prior = root/f'conditional_scale_L{args.levels}_freeze_01'
    model = picket_fence(args.levels, 2.)
    target = np.load(prior/'training.npz')['state']
    common_saved = np.load(prior/'target_common.npz')['coordinates']
    common = ConditionalDynamics(model, False)
    common_state = common.state_and_blocks(common_saved)[0]
    c, z = common.unpack(common_saved)
    raw = excitation_features(model)@z[0]
    manifest = {'L': model.L, 'train_time': 1., 'train_G': 2., 'holdout_G': [1.45, 2.55, -1.7],
                'kicks': {'none': 0.}, 'elapsed': [0., .25, .5, 1., 1.5, 2.], 'records': [],
                'protocol': 'Size replication of the identical-initial-state control. Families share the common-orbital training approximation; no future fitting.',
                'primary_comparison': 'count TV and energy-resolved pair RMS at elapsed=1'}
    np.savez_compressed(args.output/'training.npz', state=target)
    for family in ('collective_0', 'collective_1', 'collective_3', 'two_body', 'conditional_two_body', 'conditioned'):
        name = 'shared_'+family
        if family == 'conditioned':
            dynamics = ConditionalDynamics(model, True)
            coordinates = np.r_[c, np.repeat(z, model.r+1, axis=0).ravel()]
            extra = {'type': 'conditional', 'conditioned': True,
                     'complex_cone_dimension': model.r+1+(model.r-1)*(model.L-2)}
        else:
            dynamics = JastrowDynamics(model, family)
            theta, error = dynamics.project_logarithms(raw)
            assert error < 1e-10
            coordinates, _ = dynamics.encode(theta, common_state)
            extra = {'type': 'jastrow', 'jastrow_family': family, 'complex_cone_dimension': len(coordinates)}
        state = dynamics.state_and_blocks(coordinates)[0]
        difference = float(np.linalg.norm(state-common_state))
        assert difference < 1e-11
        path = args.output/(name+'.npz')
        if family == 'conditioned':
            np.savez_compressed(path, coordinates=coordinates)
        else:
            np.savez_compressed(path, coordinates=coordinates, features=dynamics.features)
        manifest['records'].append({'name': name, 'file': path.name, **extra,
                                    'initial_fidelity': float(abs(np.vdot(target, state))**2),
                                    'shared_initial_state_difference': difference,
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'L': model.L, 'families': len(manifest['records']),
                      'max_shared_initial_state_difference': max(r['shared_initial_state_difference'] for r in manifest['records'])}))
