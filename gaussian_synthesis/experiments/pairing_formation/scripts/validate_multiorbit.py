#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.multiorbit import MultiOrbitDynamics
from pairing_formation.interband import filtered_density, tomography


def job(task):
    family, L, output = task
    root = Path(__file__).resolve().parents[1]/'results'
    frozen = root/('interband_freeze_01' if L == 12 else f'interband_scale_L{L}_freeze_01')
    predictions = root/('interband_predictions_01' if L == 12 else f'interband_scale_L{L}_predictions_01')
    manifest = json.loads((frozen/'manifest.json').read_text())
    model = picket_fence(L, 2.65)
    dynamics = MultiOrbitDynamics(model, 2, svd_cutoff=1e-10)
    initial = np.load(frozen/(family+'.npz'))['coordinates'].copy()
    alpha = np.pi/2 if L == 12 else 0.
    c, logs = dynamics.unpack(initial)
    c *= np.exp(-1j*alpha*np.arange(model.r+1))[:, None]
    states, coordinates, report = dynamics.evolve(initial, manifest['elapsed'], rtol=2e-10, atol=2e-12, max_step=.02)
    name = family+'_G2.65_'+('collective_half' if alpha else 'none')
    original = np.load(predictions/(name+'.npz'))['states']
    f = manifest['filters']['target']
    differences = []
    for a, b in zip(states, original):
        ra, pa = filtered_density(model, a, f['particle'], f['hole'])
        rb, pb = filtered_density(model, b, f['particle'], f['hole'])
        differences.append(abs(tomography(ra, pa)['negativity']-tomography(rb, pb)['negativity']))
    report.update(family=family, L=L, max_state_difference=float(np.max(np.linalg.norm(states-original, axis=1))),
                  max_filtered_negativity_difference=float(max(differences)))
    np.savez_compressed(output/(name+'.npz'), states=states, coordinates=coordinates)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=int, choices=[12, 16], default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    families = ['target_two_orbits', 'witness_two_orbits'] if args.levels == 12 else ['target_two_orbits']
    records = []
    with ProcessPoolExecutor(max_workers=len(families)) as pool:
        for future in as_completed([pool.submit(job, (f, args.levels, args.output)) for f in families]):
            r = future.result(); records.append(r)
            print(json.dumps({k: r[k] for k in ('family', 'L', 'max_state_difference', 'max_filtered_negativity_difference', 'max_energy_drift')}), flush=True)
    (args.output/'validation.json').write_text(json.dumps(records, indent=2))
