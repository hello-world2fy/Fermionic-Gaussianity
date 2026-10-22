"""Audit saved closed-loop artifacts without rerunning benchmark timings."""
import bootstrap
from pathlib import Path
import csv
import argparse
import hashlib
import json
import platform
import sys
import numpy as np
from model import Anderson, gaussian_defect
from exact_solver import exact_k2


def main():
    root = Path(__file__).resolve().parent
    repository = root.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=root/'results/publication_verification.json')
    args = parser.parse_args()
    checked = 0
    max_atom_defect = 0.
    max_moment_error = 0.
    for directory in ('comparison_final', 'controls'):
        for path in sorted((root/'results'/directory).glob('case_*.json')):
            saved = json.loads(path.read_text())
            s, events = saved['summary'], saved['events']
            config = s['config']
            arrays = np.load(str(path.with_suffix(''))+'_states.npz')
            states = arrays['snapshots']
            with path.with_suffix('.csv').open() as handle:
                rows = [{key: float(value) if value else None for key, value in row.items()}
                        for row in csv.DictReader(handle)]
            model = Anderson(interaction=s['interaction'], dt=s['dt'],
                             impurity_energy=s['impurity_energy'])
            np.testing.assert_allclose(states[0], model.initial, atol=1e-14)
            if 'atoms' in arrays:
                atoms, coefficients = arrays['atoms'], arrays['coefficients']
                np.testing.assert_allclose(np.linalg.norm(atoms, axis=0), 1., atol=1e-11)
                defect = gaussian_defect(atoms)
                max_atom_defect = max(max_atom_defect, defect)
                assert defect < 1e-10, (path, defect)
                np.testing.assert_allclose(atoms @ coefficients, states[-1], atol=1e-12)
                cost = np.sum(np.abs(coefficients))**2/np.vdot(states[-1], states[-1]).real
                np.testing.assert_allclose(cost, s['final_coefficient_cost'], rtol=1e-13)
            event_by_step = {e['step']: e for e in events}
            for step in range(1, len(states)):
                if config['method'] == 'exact':
                    continue
                target = model.strang @ states[step-1]
                if step not in event_by_step:
                    np.testing.assert_allclose(states[step], target, atol=2e-12)
                    continue
                event = event_by_step[step]
                np.testing.assert_allclose(np.linalg.norm(states[step]-target),
                                           event['local_residual'], atol=2e-12)
                if 'witness' in event:
                    witness = np.array(event['witness'])
                    u = witness[:, 0]+1j*witness[:, 1]
                    moment = exact_k2(u)[1]
                    np.testing.assert_allclose(np.linalg.norm(moment-target),
                                               event['continuous_residual'], atol=2e-12)
                    if config['discretization'] == 'two_node':
                        err = np.linalg.norm(states[step]-moment)
                        max_moment_error = max(max_moment_error, float(err))
                        assert err < 2e-12, (path, step, err)
                if config['method'] == 'measure' and config['backend'].startswith('api'):
                    assert event['source_function'] == 'gaussian_synthesis.api.synthesize'
                    assert event['api_summary']['config']['orders'] == [2]
                    assert event['training_samples'] > 0 and event['validation_samples'] > 0
            if config['method'] != 'exact':
                assert all(row['compression_distance'] <= row['cumulative_residual']+2e-11
                           for row in rows), path
            if directory == 'comparison_final':
                assert abs(s['total_seconds']-s['setup_seconds']-
                           sum(s['timing_components'].values())) < 1e-12, path
                assert s['timing_components']['other_loop_overhead'] >= 0
                if config['method'] == 'measure':
                    assert s['original_solver_calls'] == 10
            checked += 1
    comparison = json.loads((root/'results/comparison_final/comparison.json').read_text())
    failed = {g['label'] for g in comparison if not g['matched_error_pass_all_seeds']}
    assert failed == {'weight', 'newton:sample_raw', 'sgd:two_node'}, failed
    source_paths = list((repository/'gaussian_synthesis/src/gaussian_synthesis').glob('*.py'))
    source_paths += [root/'vendor/four_mode_landscape/exact_solver.py',
                     root/'vendor/four_mode_landscape/_bootstrap.py']
    source_paths += list(root.glob('*.py'))
    audit = {'saved_cases_checked': checked, 'status': 'passed',
             'max_final_atom_quadric_defect': max_atom_defect,
             'max_two_node_first_moment_error': max_moment_error,
             'expected_accuracy_failures': sorted(failed),
             'python': sys.version, 'executable': sys.executable,
             'numpy': np.__version__, 'platform': platform.platform(),
             'architecture': platform.machine(),
             'sha256': {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(source_paths)}}
    args.output.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
