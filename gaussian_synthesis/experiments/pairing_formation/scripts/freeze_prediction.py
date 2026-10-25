#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.linear import squeezed_pair_matrix
from pairing_formation.frozen import channel_basis


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--replica', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = picket_fence(12, 2.)
    training_metadata = json.loads(args.replica.with_suffix('.json').read_text())
    if (training_metadata['L'], training_metadata['G'], training_metadata['time']) != (12, 2., 1.):
        raise ValueError('This protocol requires the L12 G2 t1 training witness')
    with np.load(args.replica) as data:
        state, witness = data['target'], data['witness']
    train_time = 1.
    _, ids = model.blocks[1]
    source = {'target_m1': state[ids], 'replica2_m1': witness[ids],
              'linear_rpa': squeezed_pair_matrix(model, np.array([train_time]))[0]}
    manifest = {'L': 12, 'train_G': 2., 'train_time': train_time,
                'holdout_G': [2., 2.2], 'holdout_times': [1., 1.25, 1.5, 2., 2.15, 2.5, 3.],
                'selection': 'all predeclared sources and q=1,2,3; no holdout-based choice',
                'canonical_measure_sampled': False,
                'purpose': 'diagnose frozen collective spans before canonical atom extraction',
                'source_sha256': hashlib.sha256(args.replica.read_bytes()).hexdigest(), 'bases': []}
    np.savez_compressed(args.output/'training_state.npz', state=state)
    for name, T in source.items():
        for q in (1, 2, 3):
            basis, singular = channel_basis(model, T, q)
            coefficients = basis.conj().T@state
            initial_weight = float(np.vdot(coefficients, coefficients).real)
            coefficients /= np.sqrt(initial_weight)
            filename = f'{name}_q{q}.npz'
            np.savez_compressed(args.output/filename, basis=basis, initial=coefficients)
            record = {'name': name, 'q': q, 'dimension': basis.shape[1],
                      'initial_projection_fidelity': initial_weight,
                      'T_singular_values': singular.tolist(), 'file': filename,
                      'sha256': hashlib.sha256((args.output/filename).read_bytes()).hexdigest()}
            manifest['bases'].append(record)
            print(json.dumps(record), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
