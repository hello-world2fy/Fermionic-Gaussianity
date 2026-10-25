#!/usr/bin/env python3
"""Training-only connected two-transfer geometries, followed by Hamiltonian DF.

This initial experiment uses previously inspected uniform-BCS conditions and is
therefore a pilot. A successful canonical mechanism still needs new-condition
predictions, full-measure checks and stronger alternatives.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.linear import squeezed_pair_matrix,rank_truncate,coupled_cluster_state
from pairing_formation.pair_cc_dynamics import PairCCDynamics
from pairing_formation.cluster_dynamics import ClusterDynamics,connected_two_transfer,connected_source,orthogonal_cluster_modes
from pairing_formation.sectors import transfer_operator,resolved_observables

TIMES=np.linspace(0.,2.6,53)


def train(output):
    root=Path(__file__).resolve().parents[1]/'results'
    model=picket_fence(12,2.)
    data=np.load(root/'replica2_L12_G2_01/t1.npz')
    T=squeezed_pair_matrix(model,np.array([1.]))[0]
    one,_=rank_truncate(T,1);radial=coupled_cluster_state(model,one)[model.blocks[2][1]]
    radial/=np.linalg.norm(radial)
    cc=PairCCDynamics(model);state,_,diagnostic=cc.evolve(np.array([1.]),rtol=2e-10,atol=2e-12)
    tensors={'direct':connected_two_transfer(model,data['target']),
             'witness':connected_two_transfer(model,data['witness']),
             'Hamiltonian':connected_source(model,state[0])}
    arrays={'common_radial':radial,'training_target':data['target'],'training_witness':data['witness'],
            'known_CC_training_state':state[0]}
    report={}
    for method,tensor in tensors.items():
        modes,s=orthogonal_cluster_modes(tensor,radial,4)
        arrays[method+'_modes']=np.array(modes);arrays[method+'_tensor']=tensor
        report[method]={'residual_singular_values':s.tolist(),'tensor_norm':float(np.linalg.norm(tensor)),
                        'radial_component':float(abs(np.vdot(radial,tensor))),
                        'orthogonal_mode_gram_error':float(np.linalg.norm(np.array([m.ravel() for m in modes])@np.array([m.ravel() for m in modes]).conj().T-np.eye(len(modes))))}
    np.savez_compressed(output/'frozen_training.npz',**arrays)
    (output/'training.json').write_text(json.dumps({'L':12,'G':2.,'time':1.,'profiles':report,
      'common_reference':'All methods have a full moving pair-single matrix and free complex coefficients for every transfer-count sector. A shared radial order-two cluster is derived from the leading Riccati mode.',
      'known_Hamiltonian_profile':'Exact instantaneous connected-cluster source evaluated on the conventional variational pair-CC prediction at training time.',
      'witness_scope':'Full-Gaussian k2 witness coefficients, not canonical measure samples.',
      'CC_training_integration':diagnostic},indent=2))
    digest=hashlib.sha256((output/'frozen_training.npz').read_bytes()).hexdigest()
    (output/'training_hash.json').write_text(json.dumps({'file':'frozen_training.npz','sha256':digest}))


def job(task):
    method,q,output=task;start=time.monotonic()
    model=picket_fence(12,2.);training=np.load(output/'frozen_training.npz')
    clusters=[] if method=='radial_cc' else [(2,training['common_radial'])]
    if method in ('direct','witness','Hamiltonian'):
        clusters += [(2,x) for x in training[method+'_modes'][:q]]
    dynamics=ClusterDynamics(model,clusters,radial=True,metric_cutoff=1e-10)
    states,coordinates,report=dynamics.evolve(TIMES,rtol=3e-8,atol=3e-10,max_step=.04)
    tag=f'{method}_q{q}'
    np.savez_compressed(output/f'{tag}_prediction.npz',times=TIMES,states=states,coordinates=coordinates)
    (output/f'{tag}_diagnostics.json').write_text(json.dumps(report,indent=2))
    summary={'method':method,'q':q,'seconds':time.monotonic()-start,'calls':report['calls'],
             'max_norm_error':report['max_norm_error'],'max_energy_drift':report['max_energy_drift']}
    print(json.dumps(summary),flush=True);return summary


def observables(model,states):
    operator=transfer_operator(model);rows=[]
    for time,raw in zip(TIMES,states):
        state=raw/np.linalg.norm(raw);obs=resolved_observables(model,state,operator)
        C=state[model.blocks[2][1]];singular=np.linalg.svd(C,compute_uv=False);p=float(np.sum(singular**2))
        obs.update({'time':float(time),'m2_probability':p,'m2_negativity':float((singular.sum()**2/p-1)/2) if p>1e-14 else None,
                    'energy':float(np.vdot(state,model.H@state).real)})
        rows.append(obs)
    return rows


def evaluate(output):
    source=json.loads((output/'training_hash.json').read_text())
    assert hashlib.sha256((output/source['file']).read_bytes()).hexdigest()==source['sha256']
    files=sorted(output.glob('*_prediction.npz'));assert len(files)==11
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (output/'frozen_forecasts.json').write_text(json.dumps(manifest,indent=2))
    model=picket_fence(12,2.);exact=model.evolve(TIMES)
    np.savez_compressed(output/'exact.npz',times=TIMES,states=exact)
    reference=observables(model,exact);result=[]
    for path in files:
        data=np.load(path);predicted=data['states'];rows=observables(model,predicted)
        for row,truth,x,y in zip(rows,reference,predicted,exact):
            x=x/np.linalg.norm(x);row['fidelity']=float(abs(np.vdot(x,y))**2)
            row['count_TV']=float(.5*np.sum(abs(np.array(row['count'])-truth['count'])))
            row['current_L1_error']=float(np.sum(abs(np.array(row['current'])-truth['current'])))
            A=x[model.blocks[2][1]];B=y[model.blocks[2][1]]
            row['m2_conditional_fidelity']=float(abs(np.vdot(A,B))**2/(np.linalg.norm(A)**2*np.linalg.norm(B)**2)) if min(np.linalg.norm(A),np.linalg.norm(B))>1e-10 else None
        name=path.stem.removesuffix('_prediction')
        record={'name':name,'rows':rows};result.append(record)
        primary=rows[43]
        print(json.dumps({'name':name,'time':primary['time'],**{k:primary[k] for k in ('fidelity','count_TV','current_L1_error','m2_conditional_fidelity','m2_negativity')}}),flush=True)
    (output/'evaluation.json').write_text(json.dumps({'classification':'Previously inspected-condition pilot; not a new-parameter claim.',
      'exact':reference,'models':result,'max_initial_state_difference':max(float(np.linalg.norm(np.load(p)['states'][0]-model.initial)) for p in files)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['train','predict','evaluate'],required=True)
    p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if args.stage=='train':
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps({'classification':'Bounded pilot on original uniform BCS, L12 G2. Training at t1; primary comparison t2.15. These physical conditions have already been inspected in earlier rounds.',
          'models':['radial pair-CC','radial pair-CC plus common collective two-transfer cluster','direct/witness/Hamiltonian-source extra connected two-transfer modes q=1,2,4'],
          'same_initial_state':'Exact Fermi sea at t0 for all models; only training-time geometry differs.',
          'radial_control':'Every transfer-count sector has an independent complex multiplier, so all count-sector Ehrenfest equations are represented in the tangent space, subject to the declared numerical cutoff.',
          'no_future_refits':True,'new_forecast_gate':'Only if a canonical-specific improvement survives this pilot and conventional Hamiltonian-source controls, freeze a new-parameter test.'},indent=2))
        train(args.output)
    elif args.stage=='predict':
        tasks=[('radial_cc',0),('common',0)]+[(method,q) for method in ('direct','witness','Hamiltonian') for q in (1,2,4)]
        summary=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(job,(*t,args.output)) for t in tasks]):summary.append(f.result())
        (args.output/'prediction_summary.json').write_text(json.dumps(summary,indent=2))
    else:evaluate(args.output)
