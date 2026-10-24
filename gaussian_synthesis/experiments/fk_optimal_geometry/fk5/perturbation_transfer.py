"""Preregistered new-Hamiltonian predictions; no new exact target is accepted."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .io import write_json
from . import paths
from fk.model import build_model
from fk.gaussian import sample_gaussian
from fk4.variational import predict,optimize_gaussian_energy
from fk3.discovery import optimize_gaussian_overlap


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_hamiltonian(model,t,kind,amplitude,pattern):
    H=model.full_hamiltonian(t).copy()
    if kind=='hopping':
        for a,r in enumerate(pattern):H+=.5j*t*amplitude*r*(model.gammas[a]@model.gammas[a+8])
    elif kind=='interaction':
        for x in range(2):
            for a in range(7):H-=amplitude*pattern[x][a]*(model.q_ops[x,a]@model.q_ops[x,a])
    else:raise ValueError(kind)
    assert np.max(abs(H-H.conj().T))<1e-12
    return H


def adaptive_columns(H,starts,seed,maximum=32):
    """Energy-residual pricing followed by actual-ket Ritz fits, without ED.

    The existing full-Gaussian overlap oracle prices the Ritz residual. This
    is a local energy column heuristic, not a global dictionary certificate.
    """
    rng=np.random.default_rng(seed);atoms=starts[:1].copy();history=[];snapshots={}
    for size in range(1,maximum+1):
        fit,arrays=predict(atoms,H);psi=arrays['state'];residual=H@psi-fit['energy']*psi
        history.append(dict(terms=len(atoms),energy=fit['energy'],residual_norm=float(np.linalg.norm(residual))))
        if size in [16,32]:snapshots[size]=(atoms.copy(),arrays['coefficients'].copy())
        if size==maximum:break
        if np.linalg.norm(residual)<1e-10:
            for R in [16,32]:
                if R>=size:snapshots[R]=(atoms.copy(),arrays['coefficients'].copy())
            break
        fresh=sample_gaussian(8,8,rng).kets
        candidates,search=optimize_gaussian_overlap(residual/np.linalg.norm(residual),np.concatenate([starts[:8],fresh]),max_sweeps=40,gradient_tolerance=1e-8)
        order=np.argsort(abs(candidates.conj()@residual))[::-1]
        candidates=[candidates[i] for i in order if max(abs(atoms.conj()@candidates[i]))<1-1e-10][:8]
        if not candidates:raise ArithmeticError('Residual column pricing produced no distinct Gaussian')
        trials=[]
        for candidate in candidates:
            expanded=np.concatenate([atoms,candidate[None,:]]);trial,_=predict(expanded,H)
            trials.append((trial['energy'],candidate))
        atoms=np.concatenate([atoms,min(trials,key=lambda x:x[0])[1][None,:]])
        history[-1]['pricing_seconds']=search['elapsed_seconds']
    return snapshots,history


def predict_case(output,case):
    output=Path(output);model=build_model();folder=output/'predictions'/case['key'];folder.mkdir(parents=True,exist_ok=False)
    registration=json.loads((output/'preregistration.json').read_text());freeze=np.load(output/'frozen_inputs.npz')
    t=case['t'];rep=case['pattern_id'];tag=f't{t:g}';Hfull=new_hamiltonian(model,t,case['kind'],case['amplitude'],case['pattern'])
    H=Hfull[np.ix_(model.even_masks,model.even_masks)];records=[];saved={'hamiltonian_full':Hfull}
    started=time.monotonic()
    def retain(method,R,atoms,coeff,seconds,metadata):
        name=f'{method}_R{R}';state=coeff@atoms;norm=np.linalg.norm(state);coeff=coeff/norm;state=state/norm
        saved[name+'_atoms']=atoms;saved[name+'_coefficients']=coeff
        records.append(dict(method=method,requested_maximum_terms=R,actual_terms=len(atoms),
            active_terms_above_1e_9=int(np.sum(abs(coeff)>1e-9)),coefficient_cost=float(sum(abs(coeff))),
            predicted_energy=float(np.vdot(state,H@state).real),prediction_seconds=seconds,
            uses_new_exact_ground_state=False,**metadata))
    for method in ['two_orbit','physics_prior','dual_columns','canonical_Lambda_k8','canonical_Lambda_k16','canonical_Lambda_k32']:
        source=f'{tag}_rep{0 if method=="two_orbit" else rep}_{method}'
        atoms=freeze[source]
        for R in [16,32]:
            tick=time.monotonic();fit,arrays=predict(atoms[:R],H)
            retain(method,R,atoms[:R],arrays['coefficients'],time.monotonic()-tick,
                   dict(frozen_dictionary_key=source,source_cost=registration['dictionary_sources'][source]))
    seed=int(case['seed']);tick=time.monotonic();starts=freeze[f'{tag}_pattern{rep}_HFB_starts']
    hfb,search=optimize_gaussian_energy(H,starts,sweeps=100,tolerance=1e-8);order=np.argsort(search['energies']);hfb=hfb[order]
    hfb_seconds=time.monotonic()-tick
    retain('HFB',1,hfb[:1],np.ones(1,complex),hfb_seconds,
           dict(starts=len(starts),sweeps=search['sweeps'],best_gradient=float(search['gradient_max'][order[0]]),
                converged_starts=int(sum(search['converged'])),local_optimizer=True))
    tick=time.monotonic();columns,history=adaptive_columns(H,hfb,seed,maximum=32);seconds=time.monotonic()-tick
    for R,(atoms,coeff) in columns.items():
        retain('adaptive_energy_columns',R,atoms,coeff,seconds,
               dict(HFB_initialization_seconds=hfb_seconds,energy_residual_pricing=True,
                    global_pricing_certificate=False,cost_scope='Full shared 32-term adaptive run, plus separately recorded HFB initialization'))
    write_json(folder/'column_history.json',history)
    np.savez_compressed(folder/'predictions.npz',**saved)
    result=dict(**case,rows=records,prediction_file_sha256=sha(folder/'predictions.npz'),elapsed_seconds=time.monotonic()-started,
                preregistration_sha256=sha(output/'preregistration.json'),new_target_accessed=False)
    write_json(folder/'summary.json',result)
    print(case['key'],'predicted',len(records),'in',result['elapsed_seconds'],'seconds',flush=True)
    return result
