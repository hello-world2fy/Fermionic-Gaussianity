"""Bounded L=4 baseline orchestration; no general Gaussian synthesis here."""

import numpy as np
from gaussian_synthesis import fidelity
from .model import uniform_model
from .asymptotic import asymptotic_state
from .exact import evolve_constant, evolve_inverse_time
from .mean_field import evolve_mean_field, project_spinors
from .pbcs import fit_pbcs
from .minors import minor_certificate
from .observables import observables


def run_baselines(config, budget, progress=print):
    if (config['L'], config['m']) != (4, 2):
        raise ValueError('This first pilot runner is explicitly limited to L=4,m=2')
    if abs(config['lambda']-config['L']*config['coupling']/config['bandwidth']) > 1e-12:
        raise ValueError('lambda must equal L*G/W')
    model = uniform_model(config['L'], config['m'], config['bandwidth'])
    basis = model.basis
    times = np.linspace(0, config['time_max'], config['time_points'])
    fit_indices = []
    for time in config['fit_times']:
        matches = np.flatnonzero(np.isclose(times, time, atol=1e-13, rtol=0))
        if len(matches) != 1:
            raise ValueError('Each fit time must occur exactly once on the output grid')
        fit_indices.append(int(matches[0]))
    arrays = {'times': times, 'energies': model.energies, 'pair_masks': np.array(basis.pair_masks),
              'full_masks': np.array(basis.sector.occupations), 'fit_indices': np.array(fit_indices)}
    arrays['exact'] = evolve_constant(model, model.dicke(), config['coupling'], times)
    arrays['free_exact'] = evolve_constant(model, model.dicke(), 0., times)
    for name in ('td_bcs', 'product_tdvp'):
        budget.check()
        spinors = evolve_mean_field(model, config['coupling'], times, convention=name)
        arrays[name+'_spinors'] = spinors
        arrays[name] = np.array([project_spinors(basis, s) for s in spinors])
        free = evolve_mean_field(model, 0., times, convention=name)
        arrays['free_'+name] = np.array([project_spinors(basis, s) for s in free])
        tight = evolve_mean_field(model, config['coupling'], times, convention=name, rtol=1e-12, atol=1e-14)
        arrays[name+'_tight'] = np.array([project_spinors(basis, s) for s in tight])
    for name in ('exact', 'td_bcs', 'product_tdvp'):
        data = [observables(basis, psi) for psi in arrays[name]]
        for key in data[0]:
            arrays[name+'_'+key] = np.array([row[key] for row in data])
    roots = np.random.SeedSequence(config['seed']).spawn(2)
    rngs = [np.random.default_rng(root) for root in roots]
    records = {'rng_spawn_keys': [list(root.spawn_key) for root in roots], 'quench': [], 'asymptotic': []}
    quench_rows, fits, fitted_spinors = [], [], []
    previous = None
    for index in fit_indices:
        budget.check()
        seeds = [arrays[name+'_spinors'][index] for name in ('td_bcs', 'product_tdvp')]
        if previous is not None:
            seeds.append(previous)
        fit_seed = int(rngs[0].integers(0, 2**32))
        fit = fit_pbcs(basis, arrays['exact'][index], seeds=seeds, seed=fit_seed,
                       starts=config['fit_starts'], max_seconds=min(config['fit_max_seconds'],
                                                                    budget.max_seconds-budget.elapsed))
        previous = fit.spinors
        fits.append(fit.state)
        fitted_spinors.append(fit.spinors)
        cert = minor_certificate(basis, arrays['exact'][index])
        quench_rows.append({'time': float(times[index]),
                            'F_TD': fidelity(arrays['exact'][index], arrays['td_bcs'][index]),
                            'F_TDVP': fidelity(arrays['exact'][index], arrays['product_tdvp'][index]),
                            'F_PBCS_found': fit.best_found_fidelity, 'F_PBCS_upper': cert['upper_bound'],
                            'minor_q': cert['q'], 'max_minor': cert['max_minor'],
                            'fit_seconds': fit.elapsed_seconds, 'fit_status': fit.status})
        records['quench'].append({'time': float(times[index]), 'seed': fit_seed, 'restarts': fit.restarts})
        progress(f"quench t={times[index]:g}: TD={quench_rows[-1]['F_TD']:.6f}, PBCS={fit.best_found_fidelity:.6f}")
    arrays['fitted_states'], arrays['fitted_spinors'] = np.array(fits), np.array(fitted_spinors)
    asym_rows, asym_states, asym_fits, asym_spinors = [], [], [], []
    for eta in config['etas']:
        alphas = config['alpha_scan'] if eta == config['alpha_scan_eta'] else [0., 1.]
        z = np.exp(-np.pi*eta/basis.L*np.arange(1, basis.L+1))
        onsite = np.column_stack((np.ones(basis.L), z))/np.sqrt(1+z**2)[:, None]
        for alpha in alphas:
            budget.check()
            state = asymptotic_state(model, eta, alpha=alpha)
            fit_seed = int(rngs[1].integers(0, 2**32))
            fit = fit_pbcs(basis, state, seeds=[onsite], seed=fit_seed, starts=config['fit_starts'],
                           max_seconds=min(config['fit_max_seconds'], budget.max_seconds-budget.elapsed))
            cert = minor_certificate(basis, state)
            asym_rows.append({'eta': eta, 'alpha': alpha, 'F_PBCS_found': fit.best_found_fidelity,
                              'F_PBCS_upper': cert['upper_bound'], 'minor_q': cert['q'],
                              'max_minor': cert['max_minor'], 'entropy': observables(basis, state)['entropy'],
                              'fit_seconds': fit.elapsed_seconds, 'fit_status': fit.status})
            asym_states.append(state)
            asym_fits.append(fit.state)
            asym_spinors.append(fit.spinors)
            records['asymptotic'].append({'eta': eta, 'alpha': alpha, 'seed': fit_seed, 'restarts': fit.restarts})
            progress(f'asymptotic eta={eta:g}, alpha={alpha:g}: PBCS={fit.best_found_fidelity:.6f}')
    arrays['asymptotic_states'] = np.array(asym_states)
    arrays['asymptotic_fits'] = np.array(asym_fits)
    arrays['asymptotic_spinors'] = np.array(asym_spinors)
    arrays['inverse_times'] = np.array(config['inverse_times'])
    arrays['inverse_t0'] = np.array(config['inverse_t0'])
    inverse = []
    for t0 in config['inverse_t0']:
        budget.check()
        inverse.append(evolve_inverse_time(model, config['inverse_eta'], t0, config['inverse_times']))
    arrays['inverse_states'] = np.array(inverse)
    arrays['inverse_tight'] = evolve_inverse_time(model, config['inverse_eta'], min(config['inverse_t0']),
                                                config['inverse_times'], rtol=1e-12, atol=1e-14)
    arrays['inverse_asymptotes'] = np.array([asymptotic_state(model, config['inverse_eta'], time=t, frame='physical')
                                            for t in config['inverse_times']])
    budget.check()
    return arrays, quench_rows, asym_rows, records
