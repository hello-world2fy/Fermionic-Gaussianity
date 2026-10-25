#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import PairModel
from pairing_formation.multiorbit import MultiOrbitDynamics
from pairing_formation.interband import filtered_density, tomography


def job(task):
    family, output = task
    root = Path(__file__).resolve().parents[1]/'results'
    frozen = root/'orbit_response_freeze_01'
    manifest = json.loads((frozen/'manifest.json').read_text())
    record = next(r for r in manifest['records'] if r['name'] == family)
    case = next(r for r in manifest['case_groups']['unseen'] if r['name'] == 'asymmetric_G-2.2_none')
    model = PairModel(np.array(case['energies']), case['G'])
    initial = np.load(frozen/record['file'])['coordinates']
    states, coordinates, report = MultiOrbitDynamics(model, 2, svd_cutoff=1e-10).evolve(initial, manifest['elapsed'], rtol=2e-10, atol=2e-12, max_step=.02)
    prior = np.load(root/'orbit_response_unseen_predictions_01'/(family+'__'+case['name']+'.npz'))['states']
    differences, negativity = [], []
    chosen = manifest['filter']
    for a, b in zip(states, prior):
        differences.append(float(np.linalg.norm(a-b*np.exp(-1j*np.angle(np.vdot(a, b))))))
        ra, pa = filtered_density(model, a, chosen['particle'], chosen['hole'])
        rb, pb = filtered_density(model, b, chosen['particle'], chosen['hole'])
        negativity.append(abs(tomography(ra, pa)['negativity']-tomography(rb, pb)['negativity']))
    result = {'family': family, 'case': case['name'], 'max_state_difference': max(differences),
              'max_negativity_difference': max(negativity), 'integration': report}
    np.savez_compressed(output/(family+'.npz'), states=states, coordinates=coordinates)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    families = ['target_seed20261112', 'witness_seed20261112', 'response_hamiltonian_seed20261112', 'response_generators_seed20261112']
    with ProcessPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(job, [(family, args.output) for family in families]))
    (args.output/'validation.json').write_text(json.dumps(records, indent=2))
    print(json.dumps([{k:r[k] for k in ('family','max_state_difference','max_negativity_difference')} for r in records]))
