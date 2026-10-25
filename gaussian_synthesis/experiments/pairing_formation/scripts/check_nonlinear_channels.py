#!/usr/bin/env python3
"""Independent projective-CC controls and numerical refinement, before truth."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import json
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.pair_cc_dynamics import PairCCDynamics, product_parameter_map, evolve_projective
from pairing_formation.linear import coupled_cluster_state


def job(task):
    L,eta,output=task
    model,forms=angular_model(L,4.,eta)
    times=np.linspace(0.,2.5,51)
    saved,T,report=evolve_projective(model,times)
    states=np.array([coupled_cluster_state(model,x) for x in T])
    name=f'L{L}_eta{eta:g}'
    np.savez_compressed(output/f'{name}_projective.npz',times=saved,matrices=T,states=states)
    result={'L':L,'eta':eta,'projective':report}
    if L in (10,14):
        U,_=np.linalg.qr(forms)
        refinements={}
        for method,M in [('full_cc',None),('source_only_cc',product_parameter_map(U,U))]:
            d=PairCCDynamics(model,M,metric_cutoff=1e-12)
            states,coordinates,diagnostic=d.evolve(times,rtol=2e-10,atol=2e-12,max_step=.02)
            base=np.load(output.parent/'nonlinear_channels_01'/f'{name}_{method}.npz')['states']
            differences=[]
            for a,b in zip(base,states):
                # A global phase is unobservable, and the old norm drift is
                # already reported; compare normalized rays as well as vectors.
                x=a/np.linalg.norm(a);y=b/np.linalg.norm(b)
                phase=np.vdot(y,x);phase=phase/abs(phase)
                differences.append(float(np.linalg.norm(x-phase*y)))
            refinements[method]={'max_normalized_ray_difference':max(differences),
                                 'max_raw_state_difference':float(np.max(np.linalg.norm(states-base,axis=1))),
                                 'diagnostics':diagnostic}
            np.savez_compressed(output/f'{name}_{method}_refined.npz',times=times,states=states,coordinates=coordinates)
        result['refinements']=refinements
    if eta==1 and L in (10,14):
        U,_=np.linalg.qr(forms);Q=np.eye(model.r)-U@U.T
        short=np.array([.0025,.005,.01,.02])
        full,_,_=PairCCDynamics(model).evolve(short,rtol=2e-12,atol=2e-14,max_step=.002)
        _,projective,_=evolve_projective(model,short,rtol=2e-12,atol=2e-14,max_step=.002)
        coefficient=-2j/3*Q@(model.kernel[model.r:,:model.r]**3)@Q
        norm2=float(np.linalg.norm(coefficient)**2)
        rows=[]
        for t,s,T in zip(short,full,projective):
            p=float(np.linalg.norm(Q@s[model.blocks[1][1]]@Q)**2)
            pt=float(np.linalg.norm(Q@T@Q)**2)
            rows.append({'time':float(t),'cubic_probability_coefficient':norm2,
                         'DF_probability_over_leading':p/(norm2*t**6),
                         'projective_T_probability_over_leading':pt/(norm2*t**6)})
        result['short_time']=rows
    (output/f'{name}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'condition':name,'projective_complete':report['complete'],
                      'refinement_ray_difference':{k:v['max_normalized_ray_difference'] for k,v in result.get('refinements',{}).items()}}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[(10,1.),(14,1.),(18,1.),(14,.85)]
    (args.output/'protocol.json').write_text(json.dumps({'cases':cases,
      'purpose':'Hamiltonian-derived projective right-ket pCCD control and integration refinement; all run before reading new late exact references.',
      'extra_terms':'2 T_ph [sum_b J_pb* T_pb + sum_a J_ah* T_ah - J_ph* T_ph]',
      'refinement':'L10 and L14 Hermitian TDVP: tolerances /100, step /2, metric cutoff /10. L18 refinement will be separate if needed.',
      'limitation':'Normalized projective right-ket observables are not the bivariational CC left/right prescription.'},indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(job,(L,e,args.output)) for L,e in cases]):
            results.append(future.result())
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob('*.npz'))}
    (args.output/'frozen_forecasts.json').write_text(json.dumps(files,indent=2))
