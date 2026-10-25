#!/usr/bin/env python3
"""Choose linear PPT witnesses using predicted states only; no exact references."""
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
from pairing_formation.interband import filtered_density, ppt_witness


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    frozen = root/'interband_freeze_01'
    predictions = root/'interband_predictions_01'
    manifest = json.loads((frozen/'manifest.json').read_text())
    selected = manifest['filters']['target']
    records = []
    for record in manifest['records']:
        for G in manifest['holdout_G']:
            model = picket_fence(manifest['L'], G)
            for kick, alpha in manifest['kicks'].items():
                name = f"{record['name']}_G{G:g}_{kick}"
                state = np.load(predictions/(name+'.npz'))['states'][manifest['elapsed'].index(1.)]
                rho, probability = filtered_density(model, state, selected['particle'], selected['hole'])
                witness, coefficients, predicted_expectation = ppt_witness(rho)
                path = args.output/(name+'.npz')
                np.savez_compressed(path, witness=witness, coefficients=coefficients)
                records.append({'name': name, 'family': record['name'], 'G': G, 'kick': kick, 'alpha': alpha,
                                'file': path.name, 'predicted_expectation': predicted_expectation, 'predicted_probability': probability,
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {'L': 12, 'elapsed': 1., 'filter': selected, 'records': records, 'attempts_per_witness': 1000000,
              'familywise_failure_probability': .01, 'seed': 20261114,
              'protocol': 'Prediction-only PPT eigenvector gives a linear entanglement witness; randomized Pauli importance measurements; one million attempted preparations per witness; synthetic Born samples, not laboratory data'}
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'frozen_witnesses': len(records), 'attempts_per_witness': result['attempts_per_witness']}))
