#!/usr/bin/env python3
"""Freeze response controls or sector swaps without future reference access."""
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
from pairing_formation.multiorbit import MultiOrbitDynamics
from pairing_formation.orbit_response import response_basis, tangent_response_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['response', 'swaps'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    dynamics = MultiOrbitDynamics(model, 2)
    target = np.load(root/'interband_freeze_01/training.npz')['state']
    np.savez_compressed(args.output/'training.npz', state=target)
    prior = json.loads((root/'interband_freeze_01/manifest.json').read_text())
    responses, operators, _ = response_basis(model, target, 'generators')
    manifest = {'L': 12, 'train_G': 2., 'train_time': 1., 'records': [], 'case_groups': {},
                'elapsed': [0., .1, .25, .5, 1., 1.5, 2.], 'primary_time': 1.,
                'filter': prior['filters']['target'], 'mode': args.mode,
                'primary_metrics': ['absolute negativity error', 'unnormalized filtered trace distance'],
                'initial_state_control': 'For every model, also propagate its own projected initial state exactly; decompose signed negativity error into initial-state and dynamical errors.',
                'scope': 'Post-pilot mechanism/competitor tests; only the response unseen grid is prospectively frozen before new references.'}
    known = json.loads((root/'reversal_confirmation_freeze_01/manifest.json').read_text())
    manifest['case_groups']['known'] = [
        {'name': f'picket_G{G:g}_{kick}', 'G': G, 'alpha': alpha, 'kick': kick,
         'spectrum': 'picket', 'energies': model.energies.tolist()}
        for G in known['holdout_G'] for kick, alpha in known['kicks'].items()]

    def save_coordinates(name, logs, metadata):
        coordinates, fidelity = dynamics.encode(logs, target)
        path = args.output/(name+'.npz')
        np.savez_compressed(path, coordinates=coordinates)
        state = dynamics.state(coordinates)
        manifest['records'].append({'name': name, 'type': 'multiorbit', 'rank': 2, 'file': path.name,
                                    'initial_fidelity': fidelity, **metadata,
                                    'training_generator_score': tangent_response_score(dynamics, coordinates, operators),
                                    'initial_count_TV': float(np.sum(abs(np.bincount(model.m, weights=abs(state)**2-abs(target)**2, minlength=model.r+1)))/2),
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    for seed in (20261111, 20261112):
        logs = {}
        for source in ('target', 'witness'):
            data = np.load(root/'multiorbit_training_refined_01'/f'{source}_t1_seed{seed}.npz')
            _, logs[source] = dynamics.unpack(data['coordinates'])
            save_coordinates(f'{source}_seed{seed}', logs[source], {'source': source, 'seed': seed})
        if args.mode == 'swaps':
            for base, donor in (('target', 'witness'), ('witness', 'target')):
                for m in range(1, model.r):
                    hybrid = logs[base].copy(); hybrid[m] = logs[donor][m]
                    save_coordinates(f'{base}_with_{donor}_m{m}_seed{seed}', hybrid,
                                     {'source': base, 'donor': donor, 'swapped_block': m, 'seed': seed})
    if args.mode == 'response':
        selection_path = root/'orbit_response_training_01/selection.json'
        selection = json.loads(selection_path.read_text())
        manifest['training_selection_sha256'] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        manifest['training_selection'] = selection
        for selected in selection['selected']:
            path = root/'orbit_response_training_01'/selected['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == selected['sha256']
            _, logs = dynamics.unpack(np.load(path)['coordinates'])
            save_coordinates(f"response_{selected['kind']}_seed{selected['seed']}", logs,
                             {'source': 'response_'+selected['kind'], 'seed': selected['seed'],
                              'selected_weight': selected['weight']})
        for record in prior['records']:
            if record['name'] not in ('fit_conditioned', 'fit_conditional_two_body', 'linear_rpa_q3'):
                continue
            source_path = root/'interband_freeze_01'/record['file']
            destination = args.output/record['file']
            assert hashlib.sha256(source_path.read_bytes()).hexdigest() == record['sha256']
            destination.write_bytes(source_path.read_bytes())
            manifest['records'].append(record)
        holdout = selection['protocol']['new_holdout']
        energies = model.energies
        spectra = {'warped': energies+.3*energies**3,
                   'asymmetric': energies+.25*(energies**2-np.mean(energies**2))}
        manifest['case_groups']['unseen'] = [
            {'name': f'{spectrum}_G{G:g}_{kick}', 'G': G, 'alpha': alpha, 'kick': kick,
             'spectrum': spectrum, 'energies': values.tolist()}
            for spectrum, values in spectra.items() for G in holdout['G'] for kick, alpha in holdout['kicks'].items()]
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'families': len(manifest['records']), 'predictions': {k: len(v)*len(manifest['records']) for k, v in manifest['case_groups'].items()}}))
