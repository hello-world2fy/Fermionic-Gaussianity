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
from pairing_formation.jastrow import JastrowDynamics
from evaluate_conditional import phase_diagonal


def job(task):
    family, L, output = task
    root = Path(__file__).resolve().parents[1]/'results'
    frozen = root/('jastrow_freeze_01' if L == 12 else f'jastrow_scale_L{L}_freeze_01')
    predictions = root/('jastrow_predictions_01' if L == 12 else f'jastrow_scale_L{L}_predictions_01')
    manifest = json.loads((frozen/'manifest.json').read_text())
    record = next(r for r in manifest['records'] if r['name'] == family)
    saved = np.load(frozen/record['file'])
    model = picket_fence(L, 2.55)
    dynamics = JastrowDynamics(model, record['jastrow_family'], features=saved['features'], metric_cutoff=1e-12)
    eta = .35 if L == 12 else 0.
    initial = dynamics.apply_diagonal_phase(saved['coordinates'], eta*phase_diagonal(model))
    states, coordinates, report = dynamics.evolve(initial, manifest['elapsed'], rtol=2e-10, atol=2e-12, max_step=.02)
    name = family+'_G2.55_'+('spectral_plus' if eta else 'none')
    original = np.load(predictions/(name+'.npz'))['states']
    report.update(family=family, L=L,
                  max_state_difference=float(np.max(np.linalg.norm(states-original, axis=1))),
                  max_count_difference=float(max(np.max(abs(np.array(model.observe(a)['count'])-model.observe(b)['count'])) for a, b in zip(states, original))))
    np.savez_compressed(output/(name+'.npz'), states=states, coordinates=coordinates)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=int, default=12, choices=[12, 16])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    families = ('shared_two_body', 'shared_collective_3', 'shared_conditional_two_body')
    records = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(job, (f, args.levels, args.output)) for f in families]):
            record = future.result()
            records.append(record)
            print(json.dumps({k: record[k] for k in ('family', 'L', 'max_state_difference', 'max_count_difference', 'max_energy_drift')}), flush=True)
    (args.output/'validation.json').write_text(json.dumps(records, indent=2))
