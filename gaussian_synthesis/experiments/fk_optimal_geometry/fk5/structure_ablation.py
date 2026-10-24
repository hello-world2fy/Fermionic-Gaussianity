"""Actual-atom contact ablations, coherent physical metrics and cost controls."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import svd
from .paths import ROOT, PREVIOUS
from .two_orbit import two_orbit_parameters,actual_arc_atoms,actual_rotation_atoms
from .window_variational import make_dictionary,solve_budgets
from .io import write_json
from fk.model import build_model
from fk.observables import exact_metrics
from fk2.orbits import DiagonalSpin7,covariance_from_kets
from fk4.variational import predict,diverse_order


def metric(coefficients,atoms,target,H,model):
    c=np.asarray(coefficients);state=c@atoms;norm=np.linalg.norm(state)
    c=c/norm;state=state/norm
    values=exact_metrics(state,model)
    return dict(**values,fidelity=float(abs(np.vdot(target,state))**2),
                energy=float(np.vdot(state,H@state).real),
                coefficient_cost=float(sum(abs(c))),norm=float(np.linalg.norm(state)),
                active_terms_above_1e_9=int(np.sum(abs(c)>1e-9))),c


def prepare(output,t_values,replicates):
    model=build_model();initial=json.loads((ROOT/'results/initial_certificates_02/summary.json').read_text())
    baseline=PREVIOUS/'results/frozen_dictionaries_k2_k8_k16_k32_01'
    frozen=json.loads((baseline/'summary.json').read_text());assert frozen['status']=='complete'
    symmetry=DiagonalSpin7(model);tasks=[];rows=[]
    for t in t_values:
        spectrum=next(r for r in initial['spectra'] if r['t']==t)
        A1=next(r for r in initial['A1_rows'] if r['t']==t);budget=A1['cost_upper']
        parameters,a,cp,cm=two_orbit_parameters(t,spectrum['ground_energy_interval'])
        main=actual_arc_atoms(model,a);secondary=actual_rotation_atoms(model,-a)
        lam=A1['lambda_numerator']/A1['lambda_denominator'];joint=actual_arc_atoms(model,2/(lam-2))
        rng=np.random.default_rng(np.random.SeedSequence([72001,int(round(t*10000))]))
        rotations=[np.eye(16,dtype=complex)]
        for _ in range(7):
            direction=rng.normal(size=7);direction/=np.linalg.norm(direction)
            rotations.append(symmetry.rotation(direction)[0])
        def extend(atoms):
            full=np.zeros((8,256),complex);full[:,model.even_masks]=atoms;matrices=full.reshape(8,16,16)
            return np.concatenate([(u@matrices@u.T).reshape(8,256)[:,model.even_masks] for u in rotations])
        main64=extend(main);secondary64=extend(secondary);joint64=extend(joint)
        paired=np.concatenate([np.stack([main64[8*j:8*j+8],secondary64[8*j:8*j+8]],axis=1).reshape(16,128) for j in range(4)])
        pools={
            'two_orbit':(paired,dict(groups=np.tile([0,1],32),exact_16term_cost=cp+cm,source='analytic finite two-orbit construction',parameters=parameters)),
            'main_only':(main64,dict(groups=np.zeros(64,int),source='delete entire secondary orbit')),
            'secondary_only':(secondary64,dict(groups=np.ones(64,int),source='delete entire main orbit')),
            'joint_contacts':(joint64,dict(groups=np.zeros(64,int),source='globally certified joint-witness contacts',joint_lambda=lam))}
        tick=time.monotonic();low,metadata=make_dictionary(model,t,.01,random_directions=32)
        order=diverse_order(low,64)
        pools['low_energy_delta0.01']=(low[order],dict(groups=np.zeros(64,int),source='97 actual window atoms; target-independent pivoted QR',window_delta=.01,
            construction_seconds=time.monotonic()-tick,candidate_count=len(low),selected_indices=order,
            maximum_atom_energy_excess=metadata['maximum_energy_excess']))
        for name,(atoms,metadata) in pools.items():
            key=f't{t:g}_rep0_{name}';row=save_pool(output,key,t,0,name,atoms,metadata,budget,model)
            rows.append(row);tasks.append(row)
        for replicate in replicates:
            for method in ['physics_prior','dual_columns','replica_k8_P','replica_k16_P','replica_k32_P']:
                source=next(r for r in frozen['rows'] if r['t']==t and r['replicate']==replicate and r['method']==method)
                file=baseline/(source['key']+'.npz');assert hashlib.sha256(file.read_bytes()).hexdigest()==source['file_sha256']
                candidates=np.load(file)['candidate_atoms'];unique=[]
                for i,atom in enumerate(candidates):
                    if not unique or max(abs(candidates[unique].conj()@atom))<1-1e-10:unique.append(i)
                assert len(unique)>=64,(file,len(unique))
                selected=np.array(unique)[diverse_order(candidates[unique],64)]
                atoms=candidates[selected];name=method.replace('replica_','canonical_Lambda_').replace('_P','')
                metadata=dict(groups=np.zeros(len(atoms),int),source=str(file),source_sha256=source['file_sha256'],
                    source_cost_seconds=source.get('source_generation_seconds_cold',0.),
                    original_source_metadata=source,original_candidate_count=len(candidates),unique_candidate_count=len(unique),
                    selected_original_candidate_indices=selected,selection='Deduplicate Gaussian rays, then target-independent pivoted QR',
                    sampling_measure=('coefficient Lambda proportional to |overlap|^(2k-1); legacy source named P' if method.startswith('replica') else 'deterministic prior or column baseline'))
                key=f't{t:g}_rep{replicate}_{name}';row=save_pool(output,key,t,replicate,name,atoms,metadata,budget,model)
                rows.append(row);tasks.append(row)
    write_json(output/'dictionaries.json',dict(status='complete',rows=rows))
    return tasks


def save_pool(output,key,t,replicate,method,atoms,metadata,budget,model):
    norms=np.linalg.norm(atoms,axis=1);assert np.max(abs(norms-1))<1e-11
    cov=covariance_from_kets(atoms);purity=float(np.max(abs(cov@cov+np.eye(16))));assert purity<1e-10
    # Avoid counting repeated rays as additional resources.
    gram=abs(atoms.conj()@atoms.T);np.fill_diagonal(gram,0.)
    assert np.max(gram)<1-1e-10,(key,'duplicate rays')
    path=output/'dictionaries'/f'{key}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,atoms=atoms,groups=metadata.pop('groups'))
    row=dict(key=key,t=t,replicate=replicate,method=method,budget=budget,actual_distinct_atoms=len(atoms),
             max_purity_error=purity,file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**metadata)
    return row


def solve_one(output,row,R):
    output=Path(output);model=build_model();t=row['t'];key=row['key']+f'_R{R}'
    path=output/'runs'/key;path.mkdir(parents=True,exist_ok=False)
    atoms=np.load(output/'dictionaries'/(row['key']+'.npz'))['atoms'][:R]
    target=model.target(t);H=model.full_hamiltonian(t)[np.ix_(model.even_masks,model.even_masks)]
    E0=float(np.vdot(target,H@target).real);tick=time.monotonic()
    try:
        # Two unconstrained span optima establish the cost of removing the
        # budget, separate from both normalized budget-constrained solves.
        U,s,Vh=svd(atoms.T,full_matrices=False);keep=s>1e-10*s[0]
        projected=U[:,keep].conj().T@target
        cf=Vh[keep].conj().T@(projected/s[keep])
        fit,ce=predict(atoms,H)
        constrained,arrays=solve_budgets(model,atoms,t,[row['budget']])
        variants={'unconstrained_fidelity':cf,'unconstrained_energy':ce['coefficients'],
                  'budget_fidelity':arrays['budget0_fidelity_coefficients'],'budget_energy':arrays['budget0_energy_coefficients']}
        results={};saved={'atoms':atoms,'target':target}
        for name,c in variants.items():
            values,c=metric(c,atoms,target,H,model);values['energy_error']=values['energy']-E0
            results[name]=values;saved[name+'_coefficients']=c
            if name.startswith('budget'):assert values['coefficient_cost']<=row['budget']+1e-10
        np.savez_compressed(path/'states.npz',**saved)
        record=dict(t=t,method=row['method'],replicate=row['replicate'],R=R,actual_atoms=R,budget=row['budget'],
            retained_rank=int(sum(keep)),metrics=results,normalized_budget_solver=constrained[0],
            reference_metrics=dict(**exact_metrics(target,model),energy=E0),
            elapsed_seconds=time.monotonic()-tick,exact_global_energy_optimum_claimed=False,
            deletion_scope='Specific finite dictionary; only separately proved orbit/window support bounds imply universal exclusions')
        write_json(path/'summary.json',record)
        print(key,'F',results['budget_fidelity']['fidelity'],'Eerr',results['budget_energy']['energy_error'],flush=True)
        return record
    except Exception:
        import traceback
        write_json(path/'failure.json',dict(traceback=traceback.format_exc()));raise
