#!/usr/bin/env python3
"""Stage-I-only screen of relative-phase correlations in competing channels."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model, ChannelObservables, linear_matrices, linear_pair_matrix
from pairing_formation.linear import coupled_cluster_state
from pairing_formation.frozen import channel_basis
from pairing_formation.coherent_return import factor_observables


def job(task):
    L, ratio, output=task
    model,forms=angular_model(L,4.,ratio)
    obs=ChannelObservables(model,forms)
    times=np.linspace(0.,4.,81)
    states=model.evolve(times)
    B,_=channel_basis(model,forms@forms.T,2)
    projected_weight=np.sum(abs(states@B.conj())**2,axis=1)
    rows=[]
    for t,state,weight in zip(times,states,projected_weight):
        values=obs.evaluate(state)
        values.update(factor_observables(model.m,state))
        upper_density=model.pair_density(state)[model.r:,model.r:]
        rows.append({'time':float(t),**values,
                     'bare_two_channel_span_fidelity':float(weight),
                     'upper_pair_density_eigenvalues':np.linalg.eigvalsh(upper_density).tolist(),
                     'energy':float(np.vdot(state,model.H@state).real)})
    sample_times=np.array([.5,1.,2.,3.,4.])
    linear=linear_pair_matrix(model,sample_times)
    baseline=[]
    for t,T in zip(sample_times,linear):
        state=coupled_cluster_state(model,T)
        true=states[int(round(t/.05))]
        value=obs.evaluate(state)
        value.update(factor_observables(model.m,state))
        baseline.append({'time':float(t),**value,'fidelity':float(abs(np.vdot(true,state))**2)})
    values,vectors=np.linalg.eig(linear_matrices(model)[-1])
    unstable=vectors[:,values.imag>1e-8]
    supports={}
    for name,part in [('particle',unstable[:model.r]),('hole',unstable[model.r:])]:
        u,s,_=np.linalg.svd(part,full_matrices=False)
        if len(s):
            u=u[:,s>s[0]*1e-10]
            diagonal=np.sum(abs(u)**2,axis=1)
            supports[name]={'rank':u.shape[1],'effective_level_support':float(diagonal.sum()**2/np.sum(diagonal**2)),
                            'projector_diagonal':diagonal.tolist()}
    name=f'L{L}_ratio{ratio:g}'
    np.savez_compressed(output/(name+'.npz'),times=times,states=states,energies=model.energies,kernel=model.kernel,forms=forms)
    report={'L':L,'G':4.,'relative_strength':ratio,'rows':rows,'linear_lift':baseline,
            'response_eigenvalues_real':values.real.tolist(),'response_eigenvalues_imag':values.imag.tolist(),
            'unstable_subspace':supports,'bare_two_channel_span_dimension':B.shape[1],
            'scope':'Exploratory finite angular-patch model; no material, spontaneous symmetry breaking, chiral-cat or canonical-specific discovery claim.',
            'L8_disjoint_form_factor_control':bool(L==8)}
    (output/(name+'.json')).write_text(json.dumps(report,indent=2))
    selected=next(r for r in rows if r['time']==2.)
    print(json.dumps({'name':name,**{k:selected[k] for k in ('mean','normalized_phase_locking','chirality_second','chirality_binder','bare_two_channel_span_fidelity')}}),flush=True)
    return {'name':name,'L':L,'relative_strength':ratio,'max_norm_error':max(r['norm_error_before_normalization'] for r in rows),
            'energy_drift':max(abs(r['energy']-rows[0]['energy']) for r in rows),
            'max_abs_chirality_mean':max(abs(r['chirality_mean']) for r in rows)}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'scope':'Bounded stage-I exploratory extension after repeated uniform-BCS strong-baseline null results. Retain the original goal; this is not a completed replacement objective.',
              'model':'H=2 sum eps s_z - sum J_ij s_i^+ s_j^-; J=(G/L)(fx fx^T + eta fy fy^T), fx=sqrt(2) cos(2theta), fy=sqrt(2) sin(2theta), theta=pi j/r in each of two shells eps=+/-0.5.',
              'L':[8,12,16],'G':4.,'relative_strengths':[0.,.85,1.],
              'times':[0.,4.,81],'linear_probe_times':[.5,1.,2.,3.,4.],
              'diagnostics':['count moments','relative-phase fourth pair moment on the upper shell','chirality moments with symmetry-forced zero mean','pair-density spectrum','bare two-channel span fidelity'],
              'controls':['rank-one channel','15-percent coupling splitting (eta=.85)','L8 disjoint channel supports','finite-size corrected Riccati lift'],
              'failure_rules':['Negative relative-phase correlator alone does not establish a chiral cat or new phase.',
                               'Eigenvector Neff is basis dependent at a degenerate instability; report the whole unstable-subspace participation.',
                               'No expensive synthesis or claim of novelty before an actual discrepancy beyond strong baselines is identified.']}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    tasks=[(L,ratio,args.output) for L in protocol['L'] for ratio in protocol['relative_strengths']]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job,t) for t in tasks]):
            results.append(future.result())
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
