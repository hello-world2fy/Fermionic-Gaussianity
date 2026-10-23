"""Small orbit-dictionary ablation on exactly the stored Gaussian atoms."""
import numpy as np
from fk.baselines import solve_projected_atomic_l1, target_error_estimate
from fk.certificates import separation_certificate


def run_ablation(model, source_arrays, config, budget, checkpoint=None):
    atoms = source_arrays['full_atoms']
    arc_count = len(source_arrays['structured_angles'])
    parts = {
        'fock': np.arange(128),
        'fock_arc': np.arange(128+arc_count),
        'fock_haar': np.r_[np.arange(128), np.arange(128+arc_count, len(atoms))],
        'full_saved': np.arange(len(atoms)),
    }
    report = {'status': 'running', 'rows': [],
              'scope': 'Finite Spin(7) orbit dictionaries, not explicit finite Gaussian sums.',
              'evidence_level': 'numerical_evidence', 'strict_interval_certified': False}
    arrays = {}
    for t in config['t_values']:
        target = model.target(t)
        state_error = target_error_estimate(model, t, target)['target_state_error_estimate']
        schmidt = float(np.sum(np.linalg.svd(model.embed(target).reshape(16,16), compute_uv=False)))
        for name, idx in parts.items():
            budget.check()
            solved = solve_projected_atomic_l1(model.S, atoms[idx], target)
            cert = separation_certificate(solved['coefficient_l1'], solved['residual'],
                                          schmidt, state_error=state_error)
            key = f'{name}_t{t:g}'
            arrays[key+'_indices'] = idx
            arrays[key+'_coefficients'] = solved['coefficients']
            arrays[key+'_residual'] = solved['residual']
            report['rows'].append({'key':key, 't':t, 'dictionary':name,
                                   'atom_count':len(idx),
                                   'active_atoms_1e_7':solved['active_atoms_1e_7'],
                                   'solver_status':solved['solver_status'], **cert})
            if checkpoint:
                checkpoint(report, arrays)
    report['status'] = 'complete'
    return report, arrays
