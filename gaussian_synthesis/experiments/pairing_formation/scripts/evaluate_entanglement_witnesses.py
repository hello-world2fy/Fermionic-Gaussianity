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
from pairing_formation.interband import filtered_density


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.frozen/'manifest.json').read_text())
    root = Path(__file__).resolve().parents[1]/'results'
    initial = np.load(root/'interband_freeze_01/training.npz')['state']
    rng = np.random.default_rng(manifest['seed'])
    delta = manifest['familywise_failure_probability']/len(manifest['records'])
    references = {}
    records = []
    for record in manifest['records']:
        path = args.frozen/record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        data = np.load(path)
        key = (record['G'], record['alpha'])
        model = picket_fence(manifest['L'], record['G'])
        if key not in references:
            state = model.evolve([manifest['elapsed']], initial*np.exp(-1j*record['alpha']*model.m))[0]
            f = manifest['filter']; references[key] = filtered_density(model, state, f['particle'], f['hole'])
        rho, probability = references[key]
        expectation = float(np.trace(rho@data['witness']).real)
        coef = data['coefficients']; identity = coef[0, 0]
        weight = np.sum(abs(coef))-abs(identity)
        accepted = int(rng.binomial(manifest['attempts_per_witness'], probability))
        # Choosing Pauli j with probability |w_j|/sum|w|, then multiplying
        # its +/- outcome by sign(w_j), is itself a Bernoulli variable.
        plus_probability = np.clip((1+(expectation-identity)/weight)/2, 0., 1.)
        plus = int(rng.binomial(accepted, plus_probability))
        estimate = float(identity+weight*(2*plus/max(accepted, 1)-1))
        radius = float(weight*np.sqrt(2*np.log(1/delta)/max(accepted, 1)))
        records.append({**record, 'true_expectation': expectation, 'true_success_probability': probability,
                        'accepted': accepted, 'estimate': estimate, 'one_sided_hoeffding_radius': radius,
                        'upper_confidence_bound': estimate+radius, 'entanglement_certified_in_synthetic_data': bool(accepted and estimate+radius < 0),
                        'synthetic_data': True})
    (args.output/'evaluation.json').write_text(json.dumps({'records': records, 'familywise_failure_probability': manifest['familywise_failure_probability'], 'method': 'Union bound over fixed witnesses and one-sided Hoeffding for bounded importance-sampled Pauli outcomes; exact model used only for synthetic sampling'}, indent=2))
    summary = {family: sum(r['entanglement_certified_in_synthetic_data'] for r in records if r['family'] == family) for family in sorted({r['family'] for r in records})}
    print(json.dumps({'certified_out_of_nine': summary, 'synthetic_only': True}))
