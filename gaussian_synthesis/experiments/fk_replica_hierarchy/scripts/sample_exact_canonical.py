#!/usr/bin/env python3
"""Canonical P and half-integer mass with exact integer-order normalization."""
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
from scipy.stats import t as student_t
from fk4 import paths
from fk4.io import source_manifest,write_json
from fk4.cartan_sparse import CartanMoments
from fk4.reference import log_dimension
from fk4.tempering import sample_tempered
from fk4.distributions import full_distribution
from fk4.symmetry import time_reverse
from fk.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--k',type=int,required=True,choices=[1,2,4])
    p.add_argument('--canonical',type=Path,required=True)
    p.add_argument('--moments',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--t',type=float,nargs='+',default=[0,5,9,9.1,10,10.9225,11,12,20])
    p.add_argument('--count',type=int,default=32768)
    p.add_argument('--seeds',type=int,nargs='+',default=[26101,26102,26103,26104,26105])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{key:[str(v) for v in value] if key=='moments' else str(value) if isinstance(value,Path) else value for key,value in vars(args).items()})
    model=build_model()
    backends={}
    for directory in args.moments:
        proof=json.loads((directory/'proof.json').read_text())
        verification=json.loads((directory/'verification.json').read_text())
        assert proof['status']=='complete' and verification['artifact_verification_passed']
        backends[proof['k']]=CartanMoments(proof['k'],np.load(directory/'moment_matrix.npz')['matrix'])
    certificates=json.loads((args.canonical/'summary.json').read_text())
    started=time.monotonic()
    rows=[]
    for t in args.t:
        key=f't{t:g}_k{args.k}'
        frozen=np.load(args.canonical/f'{key}_frozen.npz')
        z,u,target=frozen['z'],frozen['witness'],frozen['target']
        q=float(np.vdot(z,z).real) if args.k==1 else backends[args.k].evaluate(z,hessian=False).q
        anchors={1.:np.log(np.vdot(z,z).real)-log_dimension(8,1)}
        anchors.update({float(k):np.log(b.evaluate(z,hessian=False).q)-log_dimension(8,k) for k,b in backends.items() if k<args.k})
        observations=[]
        for seed in args.seeds:
            rng=np.random.default_rng(np.random.SeedSequence([seed,601,args.k,int(round(t*10000))]))
            sampled=sample_tempered(u,args.k-.5,args.count,rng,sweeps=2,log_z_anchors=anchors)
            a=sampled.kets.conj()@u
            B=q/float(abs(a).mean())
            phases=a/abs(a)
            C=B*(phases@sampled.kets)/len(a)
            CRB=model.S@(model.S.conj().T@C)
            CRB=.5*(CRB+time_reverse(CRB))
            features,feature_report=full_distribution(sampled.kets,model.j_matrices)
            name=f'{key}_seed{seed}'
            np.savez_compressed(args.output/f'{name}.npz',P_kets=sampled.kets,aligned_phases=phases,
                                z=z,witness=u,target=target,C_raw=C,C_RB=CRB,**features)
            observation={'seed':seed,'B_ratio':B,'B_forward_diagnostic':float(np.exp(log_dimension(8,args.k)+sampled.log_z)),
                         'rho_MC_raw':float(np.linalg.norm(C-target)),'rho_MC_RB':float(np.linalg.norm(CRB-target)),
                         'P':feature_report,'sampling':sampled.diagnostics}
            observations.append(observation)
            write_json(args.output/f'{name}.json',observation)
            print(key,seed,'B',B,'rho_MC',observation['rho_MC_RB'],flush=True)
        Bs=np.array([o['B_ratio'] for o in observations])
        halfwidth=float(student_t.ppf(.975,len(Bs)-1)*Bs.std(ddof=1)/np.sqrt(len(Bs)))
        certificate=next(r for r in certificates['rows'] if r['t']==t)
        rows.append({'t':t,'k':args.k,'B_mean':float(Bs.mean()),'B_95_percent_halfwidth':halfwidth,
                     'mass_repeat_pass':len(Bs)>=5 and halfwidth/Bs.mean()<.01,
                     'independent_full_rho_upper':certificate['full_reconstruction_residual_upper'],
                     'L_certified_saved_target':certificate['L_certified_saved_target'],
                     'B_cauchy_certified_upper':certificate.get('canonical_B_cauchy_upper'),
                     'observations':observations,'B_estimate_is_strict_upper':False})
        write_json(args.output/'summary.json',{'status':'running','rows':rows})
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'elapsed_seconds':time.monotonic()-started,
                'mass_uncertainty_empirical':True,'reconstruction_certified':True,
                'next_order_gate_pass':all(r['mass_repeat_pass'] and r['independent_full_rho_upper']<1e-3 for r in rows)})


if __name__=='__main__':
    main()
