#!/usr/bin/env python3
"""Independent saved P samples, exact canonical witness and q4 normalization.

B(new)=q4(new) E_old R / E_old[|<g|new>| R],
R=(|<g|new>|/|<g|old>|)^7. Repeated ratio estimates are empirical.
No old forward normalizer or training fit is used to estimate this mass.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t
from fk4 import paths
from fk4.io import write_json,source_manifest
from fk4.cartan_sparse import CartanMoments
from fk4.symmetry import time_reverse
from fk.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--canonical',type=Path,required=True)
    p.add_argument('--exact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{k:str(v) for k,v in vars(args).items()})
    model=build_model()
    backend=CartanMoments(4,np.load(args.exact/'moment_matrix.npz')['matrix'])
    old=json.loads((args.samples/'summary.json').read_text())
    certs=json.loads((args.canonical/'summary.json').read_text())
    rows=[]
    started=time.monotonic()
    for source in old['rows']:
        t=source['t']
        key=f't{t:g}_k4'
        canonical=np.load(args.canonical/f'{key}_frozen.npz')
        z,u,target=canonical['z'],canonical['witness'],canonical['target']
        q=backend.evaluate(z,hessian=False).q
        observations=[]
        for original in source['validation_rows']:
            seed=original['seed']
            input_path=args.samples/f'{key}_validation_seed{seed}.npz'
            saved=np.load(input_path)
            kets=saved['P_all_kets']
            a=kets.conj()@u
            aold=kets.conj()@saved['witness']
            logR=7*(np.log(abs(a))-np.log(abs(aold)))
            weights=np.exp(logR-logsumexp(logR))
            B=q/float(weights@abs(a))
            phases=a/abs(a)
            C=B*((weights*phases)@kets)
            CRB=model.S@(model.S.conj().T@C)
            CRB=.5*(CRB+time_reverse(CRB))
            features=saved['P_all_features']
            np.savez_compressed(args.output/f'{key}_seed{seed}.npz',weights=weights,aligned_phases=phases,
                                C_raw=C,C_RB=CRB,P_feature_means=weights@features,z=z,target=target)
            observation={'seed':seed,'B_ratio':B,'importance_ess':float(1/(weights@weights)),
                         'source_particle_count':len(weights),'rho_MC_raw':float(np.linalg.norm(C-target)),
                         'rho_MC_RB':float(np.linalg.norm(CRB-target)),
                         'P_feature_means':(weights@features).tolist(),
                         'max_purity_error':float(saved['P_all_purity_errors'].max()),
                         'source_kets_file':str(input_path.resolve()),'source_kets_key':'P_all_kets',
                         'source_witness_sha256':source['witness_sha256'],
                         'sampling_uncertainty_certified':False}
            observations.append(observation)
            print(key,seed,'B',B,'ESS',observation['importance_ess'],flush=True)
        Bs=np.array([o['B_ratio'] for o in observations])
        halfwidth=float(student_t.ppf(.975,len(Bs)-1)*Bs.std(ddof=1)/np.sqrt(len(Bs)))
        certificate=next(r for r in certs['rows'] if r['t']==t)
        row={'t':t,'k':4,'B_mean':float(Bs.mean()),'B_95_percent_halfwidth':halfwidth,
             'mass_repeat_pass':len(Bs)>=5 and halfwidth/Bs.mean()<.01,
             'independent_full_rho_upper':certificate['full_reconstruction_residual_upper'],
             'L_certified_saved_target':certificate['L_certified_saved_target'],
             'B_cauchy_certified_upper':certificate['canonical_B_cauchy_upper'],
             'canonical_frozen_source':str(args.canonical.resolve()),'observations':observations,
             'B_estimate_is_strict_upper':False,'P_representation':'source full Gaussian kets plus saved probability weights and aligned phases'}
        rows.append(row)
        write_json(args.output/'summary.json',{'status':'running','rows':rows})
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'elapsed_seconds':time.monotonic()-started,
               'next_order_gate_pass':all(r['mass_repeat_pass'] and r['independent_full_rho_upper']<1e-3 for r in rows),
               'reconstruction_certified':True,'mass_uncertainty_empirical':True})


if __name__=='__main__':
    main()
