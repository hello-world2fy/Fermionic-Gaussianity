#!/usr/bin/env python3
"""New spectrum/coupling holdouts, using the pilot's already frozen bases."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import PairModel
from pairing_formation.frozen import propagate
from pairing_formation.coherent_return import factor_observables


def predict(task):
    record, case, source, output = task
    data=np.load(source/record['frozen_file'])
    model=PairModel(np.array(case['energies']),case['G'])
    B,c=data['basis'],data['coefficients']
    times=np.linspace(0.,2.5,51)
    states=propagate(B.conj().T@(model.H@B),c,times)@B.T
    name=case['name']+'__'+record['name']
    path=output/(name+'.npz')
    np.savez_compressed(path,times=times,states=states)
    result={'case':case,'model':record,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps({'predicted':name}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]/'results/coherent_return_pilot_01'
    models=[r for r in json.loads((source/'manifest.json').read_text())['records'] if r['q'] in (2,3)]
    for record in models:
        assert hashlib.sha256((source/record['frozen_file']).read_bytes()).hexdigest()==record['frozen_sha256']
    cases=[]
    for L in (12,16):
        x=2*(np.arange(L)+.5)/L-1
        spectra={'cubic':x+.15*x**3,'asymmetric':x+.12*(x*x-np.mean(x*x))}
        for G in (3.6,4.4):
            for spectrum,energies in spectra.items():
                cases.append({'name':f'L{L}_G{G:g}_{spectrum}','L':L,'G':G,'spectrum':spectrum,'energies':energies.tolist()})
    protocol={'scope':'Eight new physical spectrum/coupling conditions declared after the original-G pilot, before any holdout prediction or reference; all forty model trajectories saved before exact reference calculation.',
              'training_G':4.,'training_time':1.,'cases':cases,'models':models,
              'primary_elapsed':2.,'theta':.2,'dephasing_fraction':.25,
              'primary_metrics':['absolute finite-angle echo-excess error','count total variation'],
              'secondary_metrics':['25-percent collective-count-dephasing echo sign','pure-state fidelity'],
              'failure_rule':'No canonical-specific physical prediction if conventional early RPA/direct channels are comparably accurate. Keep every condition, not only positive certificate cases.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    tasks=[(m,c,source,args.output) for c in cases for m in models if m['L']==c['L']]
    predictions=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(predict,t) for t in tasks]):
            predictions.append(future.result())
    (args.output/'predictions_saved.json').write_text(json.dumps(predictions,indent=2))
    evaluations=[]
    for case in cases:
        group=[r for r in predictions if r['case']['name']==case['name']]
        model=PairModel(np.array(case['energies']),case['G'])
        training=np.load(source/group[0]['model']['frozen_file'])['training']
        times=np.load(args.output/group[0]['file'])['times']
        initial=np.column_stack([training]+[np.load(args.output/r['file'])['states'][0] for r in group])
        exact=model.evolve(times,initial)
        np.savez_compressed(args.output/(case['name']+'_references.npz'),times=times,states=exact)
        for j,record in enumerate(group):
            states=np.load(args.output/record['file'])['states']
            rows=[]
            for t,pred,ref in zip(times,states,exact):
                a=factor_observables(model.m,pred)
                b=factor_observables(model.m,ref[:,0])
                c=factor_observables(model.m,ref[:,j+1])
                for obs in (a,b,c):
                    obs['dephased_echo_excess']=.75**2*obs['echo_difference']-.04*obs['mean']
                rows.append({'elapsed':float(t),'predicted':a,'exact':b,
                             'fidelity':float(abs(np.vdot(pred,ref[:,0]))**2),
                             'count_TV':float(np.sum(abs(np.array(a['count'])-b['count']))/2),
                             'echo_error':a['echo_excess']-b['echo_excess'],
                             'initial_only_echo_error':c['echo_excess']-b['echo_excess'],
                             'dynamics_only_echo_error':a['echo_excess']-c['echo_excess'],
                             'dephased_echo_error':a['dephased_echo_excess']-b['dephased_echo_excess']})
            evaluations.append({**record,'rows':rows})
        print(json.dumps({'evaluated':case['name']}),flush=True)
    (args.output/'evaluation.json').write_text(json.dumps(evaluations,indent=2))
