"""Recompute scalar checks from saved kets; never optimize or integrate."""

import csv
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from gaussian_synthesis import fidelity, phase_aligned_distance
from .model import BCSModel
from .sectors import PairBasis
from .asymptotic import asymptotic_state
from .mean_field import project_spinors, mean_field_energy
from .minors import minor_certificate
from .observables import observables
from .provenance import digest


def verify(output, *, check_hashes=True):
    output = Path(output)
    config = json.loads((output/'config.json').read_text())
    checks = {}
    def check(name, error, threshold):
        error, threshold = float(error), float(threshold)
        checks[name] = {'error': error, 'threshold': threshold, 'passed': bool(np.isfinite(error) and error <= threshold)}
    limits = config['thresholds']
    with np.load(output/'states.npz', allow_pickle=False) as saved:
        a = {key: saved[key] for key in saved.files}
    basis = PairBasis(config['L'], config['m'])
    model = BCSModel(basis, a['energies'])
    check('energy_grid', np.max(abs(a['energies']-np.linspace(-config['bandwidth']/2, config['bandwidth']/2, basis.L))), 0)
    expected_times = np.linspace(0, config['time_max'], config['time_points'])
    if a['times'].shape != expected_times.shape:
        raise ValueError('Saved time grid has the wrong shape')
    check('time_grid', np.max(abs(a['times']-expected_times)), 0)
    if len(a['fit_indices']) != len(config['fit_times']):
        raise ValueError('Saved fit index count does not match the frozen configuration')
    check('fit_grid', np.max(abs(a['times'][a['fit_indices']]-config['fit_times'])), 1e-13)
    check('pair_mask_order', np.max(np.abs(a['pair_masks']-basis.pair_masks)), 0)
    check('full_mask_order', np.max(np.abs(a['full_masks']-basis.sector.occupations)), 0)
    state_keys = ['exact', 'free_exact', 'td_bcs', 'product_tdvp', 'fitted_states',
                  'asymptotic_states', 'asymptotic_fits', 'inverse_states', 'inverse_tight']
    check('state_norms', max(np.max(abs(np.linalg.norm(a[key], axis=-1)-1)) for key in state_keys), limits['state_norm_error'])
    check('initial_dicke', phase_aligned_distance(a['exact'][0], model.dicke()), limits['formula_or_observable_error'])
    H = model.hamiltonian(config['coupling'])
    energies = np.einsum('bi,ij,bj->b', a['exact'].conj(), H, a['exact']).real
    check('exact_energy', np.ptp(energies), limits['conserved_quantity_error'])
    expm_error = max(phase_aligned_distance(a['exact'][index], expm(-1j*a['times'][index]*H) @ model.dicke())
                     for index in a['fit_indices'])
    check('independent_matrix_exponential', expm_error, limits['formula_or_observable_error'])
    observable_errors = []
    for name in ('exact', 'td_bcs', 'product_tdvp'):
        for index, state in enumerate(a[name]):
            for key, value in observables(basis, state).items():
                observable_errors.append(np.max(np.abs(a[name+'_'+key][index]-value)))
    check('stored_observables', max(observable_errors), limits['formula_or_observable_error'])
    for name in ('td_bcs', 'product_tdvp'):
        spinors = a[name+'_spinors']
        check(name+'_spinor_norms', np.max(abs(np.sum(abs(spinors)**2, axis=2)-1)), limits['conserved_quantity_error'])
        check(name+'_mean_number', np.max(abs(np.sum(abs(spinors[:, :, 1])**2, axis=1)-basis.m)), limits['conserved_quantity_error'])
        e = [mean_field_energy(model.energies, s, config['coupling'], name) for s in spinors]
        check(name+'_own_energy', np.ptp(e), limits['conserved_quantity_error'])
        check(name+'_free_exactness', max(1-fidelity(x, y) for x, y in zip(a['free_exact'], a['free_'+name])), limits['free_infidelity'])
        check(name+'_tolerance_convergence', max(phase_aligned_distance(x, y) for x, y in zip(a[name], a[name+'_tight'])), limits['numerical_convergence_distance'])
        check(name+'_projection', max(phase_aligned_distance(x, project_spinors(basis, s)) for x, s in zip(a[name], spinors)), limits['formula_or_observable_error'])
    with (output/'baselines.csv').open() as stream:
        quench = list(csv.DictReader(stream))
    with (output/'asymptotic.csv').open() as stream:
        asym = list(csv.DictReader(stream))
    expected_asym = [(eta, alpha) for eta in config['etas']
                     for alpha in (config['alpha_scan'] if eta == config['alpha_scan_eta'] else [0., 1.])]
    if not (len(quench) == len(a['fitted_states']) == len(a['fitted_spinors']) == len(config['fit_times'])):
        raise ValueError('Missing quench rows or fitted states')
    if not (len(asym) == len(a['asymptotic_states']) == len(a['asymptotic_fits']) == len(a['asymptotic_spinors']) == len(expected_asym)):
        raise ValueError('Missing asymptotic rows or fitted states')
    if [(float(row['eta']), float(row['alpha'])) for row in asym] != expected_asym:
        raise ValueError('Asymptotic rows do not match the frozen configuration')
    scalar_error, bracket_error, nesting_error, fit_membership, phase_probability_error = [], [], [], [], []
    for row, index, fitted, spinor in zip(quench, a['fit_indices'], a['fitted_states'], a['fitted_spinors']):
        target = a['exact'][index]
        cert = minor_certificate(basis, target)
        found = fidelity(target, fitted)
        scalar_error.extend([abs(float(row['F_PBCS_found'])-found), abs(float(row['F_PBCS_upper'])-cert['upper_bound']),
                             abs(float(row['time'])-a['times'][index]),
                             abs(float(row['minor_q'])-cert['q']), abs(float(row['max_minor'])-cert['max_minor']),
                             abs(float(row['F_TD'])-fidelity(target, a['td_bcs'][index])),
                             abs(float(row['F_TDVP'])-fidelity(target, a['product_tdvp'][index]))])
        bracket_error.append(max(0., found-cert['upper_bound']))
        nesting_error.append(max(0., max(fidelity(target, a[name][index]) for name in ('td_bcs', 'product_tdvp'))-found))
        fit_membership.extend([phase_aligned_distance(fitted, project_spinors(basis, spinor)), minor_certificate(basis, fitted)['max_minor']])
    asym_formula_errors, control_errors = [], []
    for row, target, fitted, spinor in zip(asym, a['asymptotic_states'], a['asymptotic_fits'], a['asymptotic_spinors']):
        eta, alpha = float(row['eta']), float(row['alpha'])
        expected = asymptotic_state(model, eta, alpha=alpha)
        cert = minor_certificate(basis, target)
        found = fidelity(target, fitted)
        scalar_error.extend([abs(float(row['F_PBCS_found'])-found), abs(float(row['F_PBCS_upper'])-cert['upper_bound']),
                             abs(float(row['minor_q'])-cert['q']), abs(float(row['max_minor'])-cert['max_minor']),
                             abs(float(row['entropy'])-observables(basis, target)['entropy'])])
        bracket_error.append(max(0., found-cert['upper_bound']))
        fit_membership.extend([phase_aligned_distance(fitted, project_spinors(basis, spinor)), minor_certificate(basis, fitted)['max_minor']])
        asym_formula_errors.append(phase_aligned_distance(target, expected))
        control = asymptotic_state(model, eta, alpha=0)
        phase_probability_error.append(np.max(abs(abs(target)**2-abs(control)**2)))
        if alpha == 0:
            control_errors.append(1-found)
    check('recomputed_fidelities', max(scalar_error), limits['recomputed_scalar_error'])
    check('onsite_lower_upper_bracket', max(bracket_error), limits['fidelity_bracket_slack'])
    check('baseline_nesting', max(nesting_error), limits['fidelity_bracket_slack'])
    check('fitted_states_are_pbcs', max(fit_membership), limits['formula_or_observable_error'])
    check('asymptotic_formula', max(asym_formula_errors), limits['formula_or_observable_error'])
    check('phase_control_probabilities', max(phase_probability_error), limits['formula_or_observable_error'])
    check('phase_removed_pbcs_exactness', max(control_errors), limits['free_infidelity'])
    check('fit_budgets_completed', sum(row['fit_status'] != 'best_found_only' for row in quench+asym), 0)
    infidelities = [[1-fidelity(psi, reference) for psi, reference in zip(states, a['inverse_asymptotes'])]
                   for states in a['inverse_states']]
    check('inverse_final_asymptote', max(row[-1] for row in infidelities), limits['inverse_final_infidelity'])
    check('inverse_late_time_improvement', max(0., max(row[-1]-row[0] for row in infidelities)), 0)
    check('inverse_initial_cutoff', max(phase_aligned_distance(x, y) for x, y in zip(*a['inverse_states'])), limits['inverse_initial_cutoff_distance'])
    smallest = int(np.argmin(a['inverse_t0']))
    check('inverse_tolerance_convergence', max(phase_aligned_distance(x, y) for x, y in zip(a['inverse_states'][smallest], a['inverse_tight'])), limits['numerical_convergence_distance'])
    if check_hashes:
        manifest = json.loads((output/'manifest.json').read_text())
        mismatches = sum(digest(output/relative) != expected for relative, expected in manifest['artifact_sha256'].items())
        mismatches += sum(digest(output/'source_snapshot'/relative) != expected
                          for relative, expected in manifest['environment']['source_sha256'].items())
        check('saved_hashes', mismatches, 0)
    return {'status': 'passed' if all(row['passed'] for row in checks.values()) else 'failed_acceptance',
            'checks': checks, 'inverse_infidelities': infidelities,
            'scope': 'L=4 classical baselines and onsite PBCS; no general PG or Gaussian extent computed'}
