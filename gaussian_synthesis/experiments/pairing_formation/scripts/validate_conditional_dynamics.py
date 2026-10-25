#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import argparse
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.conditional_dynamics import ConditionalDynamics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--levels', type=int, default=12, choices=[8, 12, 16])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    frozen = root/('conditional_dynamics_freeze_01' if args.levels == 12 else f'conditional_scale_L{args.levels}_freeze_01')
    prediction_directory = root/('conditional_dynamics_predictions_01' if args.levels == 12 else f'conditional_scale_L{args.levels}_predictions_01')
    manifest = json.loads((frozen/'manifest.json').read_text())
    validation = []
    for family in ('target_conditioned', 'target_common'):
        record = next(x for x in manifest['records'] if x['name'] == family)
        model = picket_fence(args.levels, 2.5)
        dynamics = ConditionalDynamics(model, record['conditioned'], metric_cutoff=1e-12)
        initial = np.load(frozen/record['file'])['coordinates'].copy()
        c, z = dynamics.unpack(initial)
        eta = .3 if args.levels == 12 else 0.
        z -= 1j*eta*np.r_[2*model.energies[model.r:]-1, -2*model.energies[:model.r]-1]
        states, coordinates, report = dynamics.evolve(initial, manifest['elapsed'], rtol=2e-10, atol=2e-12, max_step=.02)
        name = family+'_G2.5_'+('spectral_plus' if eta else 'none')
        original = np.load(prediction_directory/(name+'.npz'))['states']
        report['family'] = family
        report['L'] = model.L
        report['max_state_difference'] = float(np.max(np.linalg.norm(states-original, axis=1)))
        report['max_count_difference'] = float(max(np.max(abs(np.array(model.observe(a)['count'])-model.observe(b)['count'])) for a, b in zip(states, original)))
        validation.append(report)
        np.savez_compressed(args.output/(name+'.npz'), states=states, coordinates=coordinates)
        print(json.dumps({k: report[k] for k in ('family', 'max_state_difference', 'max_count_difference', 'max_energy_drift')}))
    (args.output/'validation.json').write_text(json.dumps(validation, indent=2))
