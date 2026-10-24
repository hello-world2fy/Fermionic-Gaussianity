#!/usr/bin/env python3
"""Fixed-witness moment and transverse overlap-width scan, not changing w_k."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import source_manifest,write_json
from fk4.cartan_sparse import CartanMoments
from fk4.reference import log_dimension
from fk4.tempering import sample_tempered
from fk4.distributions import full_distribution
from fk.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixed',type=Path,required=True)
    p.add_argument('--moments',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=8192)
    p.add_argument('--t',type=int,nargs='+',default=[0,9,11,20])
    p.add_argument('--orders',type=int,nargs='+',default=[2,4,8,12,16,24,32,48,64,96,128])
    p.add_argument('--seeds',type=int,nargs='+',default=[27101,27102,27103,27104,27105])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{key:[str(v) for v in value] if key=='moments' else str(value) if isinstance(value,Path) else value for key,value in vars(args).items()})
    model=build_model()
    backends={}
    for directory in args.moments:
        proof=json.loads((directory/'proof.json').read_text())
        assert json.loads((directory/'verification.json').read_text())['artifact_verification_passed']
        backends[proof['k']]=CartanMoments(proof['k'],np.load(directory/'moment_matrix.npz')['matrix'])
    fixed=json.loads((args.fixed/'summary.json').read_text())
    rows=[]
    started=time.monotonic()
    for t in args.t:
        data=np.load(args.fixed/f't{t:g}_fixed.npz')
        z,w=data['z'],data['witness']
        geometry=next(r for r in fixed['rows'] if r['t']==t)
        h=geometry['support_value_exact'] or geometry['local_support_max']
        anchors={1.:np.log(np.vdot(z,z).real)-log_dimension(8,1)}
        anchors.update({float(k):np.log(b.evaluate(z,hessian=False).q)-log_dimension(8,k) for k,b in backends.items()})
        for seed in args.seeds:
            observations=[]
            arrays={'z':z,'witness':w}
            def snapshot(kets,beta,log_z,log_r,diagnostic):
                if beta not in args.orders:
                    return
                features,report=full_distribution(kets,model.j_matrices)
                deficit=2*np.log(h)-log_r
                row={'k':int(beta),'log_q':log_dimension(8,int(beta))+log_z,
                     'log_q_minus_2k_logh':log_dimension(8,int(beta))+log_z-2*beta*np.log(h),
                     'reference_support':h,'reference_support_is_exact':t==0,
                     'overlap_log_deficit_mean':float(deficit.mean()),
                     'overlap_width_rms':float(np.sqrt(max(0,deficit.mean()))),
                     'particles_exceeding_local_support':int(np.sum(deficit< -1e-8)),
                     'P_at_beta_k':report,'moment_anchor_used':beta in anchors}
                observations.append(row)
                key=f'k{int(beta)}'
                arrays.update({key+'_sample_kets':kets[:256].copy(),key+'_log_r':log_r.copy(),
                               key+'_features':features['features']})
            rng=np.random.default_rng(np.random.SeedSequence([seed,701,t]))
            sampled=sample_tempered(w,max(args.orders),args.count,rng,sweeps=2,checkpoints=args.orders,
                                     state_callback=snapshot,log_z_anchors=anchors)
            name=f't{t:g}_seed{seed}'
            np.savez_compressed(args.output/f'{name}.npz',**arrays)
            row={'t':t,'seed':seed,'observations':observations,'sampling':sampled.diagnostics,
                 'fixed_source':str(args.fixed.resolve()),'exploratory_geometry':True,
                 'not_a_canonical_order_scan':True}
            rows.append(row)
            write_json(args.output/f'{name}.json',row)
            write_json(args.output/'summary.json',{'status':'running','rows':rows})
            print(name,'completed through',max(args.orders),'seconds',sampled.diagnostics['elapsed_seconds'],flush=True)
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'elapsed_seconds':time.monotonic()-started,
                                        'global_support_known_only_at_t0':True})


if __name__=='__main__':
    main()
