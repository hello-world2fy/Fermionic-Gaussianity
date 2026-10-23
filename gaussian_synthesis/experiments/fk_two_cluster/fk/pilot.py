"""Fixed-budget calibration, empirical training, and held-out validation."""
import hashlib
import math
import numpy as np
from .dictionaries import Dictionary
from .distributions import CoherentBins
from .moments import ProjectedMoments
from .observables import atom_invariants
from .solver import solve
from .validation import gaussian_probe_reference,formal_budget,ReconstructionAccumulator


def stream(seed,name,purpose):
    # Fixed labels: 0=calibration, 1=training, 2=held-out final validation.
    ss = np.random.SeedSequence([int(seed),int(name=='product'),int(purpose)])
    return np.random.default_rng(ss)


def calibrate(model,name,seed,config,budget,checkpoint=None):
    dictionary = Dictionary(name)
    probe = np.zeros(128,complex); probe[0] = 1.
    orders = config['orders']
    acc = ReconstructionAccumulator(dictionary,np.tile(probe,(len(orders),1)),orders)
    rng = stream(seed,name,0)
    rows = []
    for size in config['sample_sizes']:
        while acc.count < size:
            budget.check()
            batch = dictionary.sample(min(config['batch_size'],size-acc.count),rng)
            acc.add(batch.kets)
            if checkpoint and (acc.count % 4096 == 0 or acc.count == size):
                checkpoint({'status':'partial','completed_samples':acc.count,'rows':rows,
                            'rng_state':rng.bit_generator.state,'formally_certified':False},acc.snapshot())
        for index,k in enumerate(orders):
            ref = gaussian_probe_reference(dictionary,k)
            val = acc.result(index,probe,model.S)
            qse = math.sqrt(max(0.,ref['q_second_moment']-1)/size)
            bse = math.sqrt(max(0.,ref['coefficient_second_moment']-ref['mass']**2)/size)
            crms = math.sqrt(max(0.,ref['coefficient_second_moment']-1)/size)
            passed = (abs(val['q_validation']-1) <= 5*qse+1e-12
                      and abs(val['mass_validation']-ref['mass']) <= 5*bse+1e-12
                      and val['sampled_full_residual'] <= 6*crms+1e-12)
            rows.append({'dictionary':name,'seed':seed,'k':k,'samples':size,
                         'q':val['q_validation'],'q_expected':1.,'q_standard_error_exact':qse,
                         'mass':val['mass_validation'],'mass_expected':ref['mass'],
                         'mass_standard_error_exact':bse,'sampled_full_residual':val['sampled_full_residual'],
                         'reconstruction_rmse_exact':crms,'diagnostic_calibration_passed':passed,
                         'raw_precision_target_reached':val['sampled_full_residual']<=config['reconstruction_tolerance']})
    final = [r for r in rows if r['samples']==max(config['sample_sizes'])]
    return {'status':'passed' if all(r['diagnostic_calibration_passed'] for r in final) else 'calibration_failed',
            'rows':rows,'interpretation':'5 exact standard errors for scalar moments; 6 RMS for reconstruction. Diagnostic consistency, not a confidence certificate.',
            'usage':budget.usage()}


def train(model,name,seed,config,budget,allowed_orders=None,checkpoint=None):
    dictionary = Dictionary(name)
    rng = stream(seed,name,1)
    maximum = max(config['sample_sizes'])
    h = np.empty((maximum,3),complex)
    for first in range(0,maximum,config['batch_size']):
        budget.check()
        batch = dictionary.sample(min(config['batch_size'],maximum-first),rng)
        h[first:first+len(batch.kets)] = np.einsum('ni,ij->nj',batch.kets.conj(),model.S,optimize=False)
    rows, arrays, skipped = [], {'training_overlaps':h}, []
    for size in config['sample_sizes']:
        for t in config['t_values']:
            target = model.target(t)
            reduced = np.einsum('ij,i->j',model.S.conj(),target,optimize=False)
            previous = None
            for k in config['orders']:
                if allowed_orders is not None and k not in allowed_orders:
                    skipped.append({'dictionary':name,'seed':seed,'t':t,'k':k,'train_samples':size,
                                    'status':'skipped_calibration_failed'})
                    continue
                budget.check()
                backend = ProjectedMoments(h[:size],k,dictionary.log_frame_constant(k))
                solution = solve(backend,reduced,previous,config['gradient_tolerance'],config['max_newton_steps'],budget)
                previous = solution['z']
                value = backend.evaluate(previous,hessian=False)
                key = f'n{size}_t{t}_k{k}'
                u = np.einsum('ij,j->i',model.S,previous,optimize=False)
                arrays[key+'_u'] = u
                arrays[key+'_z'] = previous
                arrays[key+'_history'] = np.array([[x['iteration'],x['objective'],x['gradient_norm']] for x in solution['history']]).reshape(-1,3)
                rows.append({'key':key,'dictionary':name,'seed':seed,'t':t,'k':k,'train_samples':size,
                             'optimizer_converged':solution['converged'],'optimizer_status':solution['status'],
                             'gradient_norm':solution['gradient_norm'],'gradient_kind':'exact_frame' if k==1 else 'empirical_projected',
                             'iterations':solution['iterations'],'q_train':value.q,'mass_train':value.mass,
                             'training_mass_ess':value.ess,
                             'witness_sha256':hashlib.sha256(u.tobytes()).hexdigest()})
                if checkpoint:
                    checkpoint({'status':'training_checkpoint','rows':rows,'skipped':skipped,'usage':budget.usage()},arrays)
    return {'status':'trained','rows':rows,'skipped':skipped,'usage':budget.usage()},arrays


def validate(model,name,seed,config,training,arrays,budget,checkpoint=None):
    dictionary = Dictionary(name)
    rows = training['rows']
    witnesses = np.array([arrays[r['key']+'_u'] for r in rows])
    orders = [r['k'] for r in rows]
    cap = max(config['sample_sizes'])
    # Reserve failure probability for all planned candidates, including skipped ones.
    candidate_count = (len(config['dictionaries'])*len(config['seeds'])*len(config['sample_sizes'])
                       *len(config['t_values'])*len(config['orders']))
    delta = config['total_failure_probability']/(2*candidate_count)
    for row,u in zip(rows,witnesses):
        row['statistical_plan'] = formal_budget(dictionary,row['k'],np.linalg.norm(u),delta,
                                               config['statistical_radius_target'],cap)
    # This held-out fixed-size diagnostic is not relabelled as a completed
    # formal certificate when the precomputed required N exceeds the cap.
    acc = ReconstructionAccumulator(dictionary,witnesses,orders)
    bins = {i:CoherentBins() for i,r in enumerate(rows)
            if r['train_samples']==cap and r['t'] in (9,11,20) and r['k'] in (1,2,4)}
    rng = stream(seed,name,2)
    while acc.count < cap:
        budget.check()
        batch = dictionary.sample(min(config['batch_size'],cap-acc.count),rng)
        coeff = acc.add(batch.kets)
        inv = atom_invariants(batch.covariances,model.j_matrices)
        for index,binner in bins.items():
            binner.add(batch.kets,coeff[:,index],inv)
        if checkpoint and (acc.count % 4096 == 0 or acc.count == cap):
            checkpoint({'status':'partial','completed_samples':acc.count,'rows':rows,
                        'rng_state':rng.bit_generator.state,'formally_certified':False},acc.snapshot())
    dist_arrays = {}
    for i,row in enumerate(rows):
        target = model.target(row['t'])
        val = acc.result(i,target,model.S)
        reconstruction = val.pop('reconstruction')
        arrays[row['key']+'_heldout_reconstruction'] = reconstruction
        row.update(val)
        raw_residual = row['sampled_full_residual']
        row['validation_full_residual'] = float(np.linalg.norm(witnesses[i]-target)) if row['k']==1 else raw_residual
        row['validation_residual_kind'] = 'analytic_frame_identity' if row['k']==1 else 'independent_mc_diagnostic'
        row['sampled_residual_pythagoras_error'] = abs(raw_residual**2-row['validation_singlet_residual']**2-row['validation_outside_singlet_norm']**2)
        row['reconstruction_accepted'] = bool(row['optimizer_converged'] and row['validation_full_residual']<=config['reconstruction_tolerance'])
        plan = row['statistical_plan']
        row['statistical_status'] = plan['status']
        row['statistically_certified'] = False
        if plan['status']=='ready':
            radius = math.sqrt(plan['second_moment_bound']/(delta*cap))
            row['formal_residual_upper'] = row['validation_full_residual'] if row['k']==1 else raw_residual+radius
            row['formal_mass_upper'] = row['mass_validation']+radius
            row['formal_repaired_cost_upper'] = row['formal_mass_upper']+math.sqrt(128)*row['formal_residual_upper']
            row['statistically_certified'] = row['formal_residual_upper'] <= config['reconstruction_tolerance']
            row['statistical_status'] = 'certified' if row['statistically_certified'] else 'not_certified'
        row['empirical_fock_repaired_cost'] = row['mass_validation']+float(np.sum(np.abs(target-reconstruction)))
        row['distribution_interpretation_allowed'] = False
        if i in bins:
            b = bins[i].result(model,target)
            row['coherent_bin_closure_residual'] = b['closure_residual']
            row['distribution_interpretation_allowed'] = bool(row['reconstruction_accepted'] and b['closure_residual']<=config['reconstruction_tolerance'])
            # Preserve raw diagnostic bins for audit even when interpretation is withheld.
            for field in ('histogram','coherent_vectors','correlation_kernel'):
                dist_arrays[row['key']+'_'+field] = b.pop(field)
            row['coherent_bin_diagnostics'] = b
    return {'status':'complete','dictionary':name,'seed':seed,'rows':rows,
            'random_streams':{'calibration':[seed,int(name=='product'),0],
                              'training':[seed,int(name=='product'),1],
                              'heldout_validation':[seed,int(name=='product'),2]},
            'failure_probability_reserved':config['total_failure_probability'],
            'planned_candidate_count':candidate_count,'usage':budget.usage()},arrays,dist_arrays
