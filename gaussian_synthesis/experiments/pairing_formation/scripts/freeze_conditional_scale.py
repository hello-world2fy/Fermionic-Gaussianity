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
from pairing_formation.fitting import fit_product_family
from pairing_formation.conditional import fit_conditional_products
from pairing_formation.conditional_dynamics import ConditionalDynamics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = picket_fence(args.levels, 2.)
    target = model.evolve([1.])[0]
    np.savez_compressed(args.output/'training.npz', state=target)
    independent, basis, x, _, _ = fit_conditional_products(model, target, starts=8, seed=20261103)
    common, _ = fit_product_family(model, target, radial=True, starts=8, seed=20261103)
    common_x = np.array(common['parameters'])
    families = [('target_conditioned', True, x[:, :model.L]+1j*x[:, model.L:]),
                ('target_common', False, (common_x[:model.L]+1j*common_x[model.L:])[None])]
    manifest = {'L': model.L, 'train_time': 1., 'train_G': 2., 'holdout_G': [1.5, 2.5, -1.8],
                'kicks': {'none': 0.}, 'elapsed': [0., .25, .5, 1., 1.5, 2.],
                'protocol': 'Size replication; training accesses only t=1, all nonlinear predictions precede exact references', 'records': []}
    for name, conditioned, logarithms in families:
        coordinates, fidelity = ConditionalDynamics(model, conditioned).encode(logarithms, target)
        path = args.output/(name+'.npz')
        np.savez_compressed(path, coordinates=coordinates)
        manifest['records'].append({'name': name, 'file': path.name, 'type': 'tdvp', 'conditioned': conditioned,
                    'complex_manifold_dimension': model.r+1+(model.r-1 if conditioned else 1)*(model.L-2),
                    'initial_fidelity': fidelity, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output/'training_fits.json').write_text(json.dumps({'conditioned': independent, 'common': common}, indent=2))
    print(json.dumps({'L': model.L, 'common': common['fidelity'], 'conditioned': independent['fidelity']}))
