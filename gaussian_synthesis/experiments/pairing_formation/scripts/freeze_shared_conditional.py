#!/usr/bin/env python3
"""Ablate count-dependent orbital motion from an identical initial state."""
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
from pairing_formation.conditional_dynamics import ConditionalDynamics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    target = np.load(root/'replica2_L12_G2_01/t1.npz')['target']
    original = np.load(root/'conditional_dynamics_freeze_01/target_common.npz')['coordinates']
    c, z = ConditionalDynamics(model, False).unpack(original)
    manifest = {'L': 12, 'train_time': 1., 'train_G': 2., 'holdout_G': [1.55, 2.45, -1.9],
                'kicks': {'none': 0., 'spectral_plus': .25, 'spectral_minus': -.25},
                'elapsed': [0., .25, .5, 1., 1.5, 2.], 'records': [],
                'protocol': 'Both manifolds start in the identical approximate physical state; only permission for count-dependent orbital motion differs; fresh parameter/kick values'}
    np.savez_compressed(args.output/'training.npz', state=target)
    states = []
    for conditioned in (False, True):
        name = 'shared_conditioned' if conditioned else 'shared_common'
        logarithms = np.repeat(z, model.r+1, axis=0) if conditioned else z.copy()
        coordinates = np.r_[c, logarithms.ravel()]
        state = ConditionalDynamics(model, conditioned).state_and_blocks(coordinates)[0]
        states.append(state)
        path = args.output/(name+'.npz')
        np.savez_compressed(path, coordinates=coordinates)
        manifest['records'].append({'name': name, 'file': path.name, 'type': 'tdvp', 'conditioned': conditioned,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'initial_fidelity': float(abs(np.vdot(target, state))**2)})
    manifest['initial_state_difference'] = float(np.linalg.norm(states[0]-states[1]))
    assert manifest['initial_state_difference'] < 1e-14
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'initial_state_difference': manifest['initial_state_difference']}))
