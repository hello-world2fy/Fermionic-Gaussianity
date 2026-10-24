#!/usr/bin/env python3
"""Freeze an independent near-optimal witness before varying replica order."""
from pathlib import Path
import argparse
import os
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import source_manifest,write_json
from fk4.exact_cartan import ExactCartan
from fk4.geometry import contact_hessian
from fk4.distributions import full_distribution
from fk.model import build_model
from fk.gaussian import sample_gaussian
from fk3.discovery import optimize_gaussian_overlap


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--columns',type=Path,required=True)
    p.add_argument('--exact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--starts',type=int,default=64)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    model=build_model()
    exact=ExactCartan(args.exact/'proof.json')
    rows=[]
    for t in [0,9,11,20]:
        started=time.monotonic()
        source=np.load(args.columns/f'columns_t{t:g}_seed25101.npz')
        z=source['dual_z'].real.astype(complex)
        certificate=exact.certificate(z,source['target'])
        z=certificate.pop('z_feasible')
        if t==0:
            z=np.array([2.,0.,0.],complex)
            support_status='h=1 from the exact two-magic Gaussian-fidelity multiplicativity theorem; not inferred from local search'
        else:
            support_status='h<=1 certified by exact q4; local maximizers give lower support estimates only'
        w=model.S@z
        rng=np.random.default_rng(27001+t)
        starts=np.concatenate((sample_gaussian(8,args.starts,rng).kets,source['generated_pricing_atoms'][-32:]))
        contacts,diagnostic=optimize_gaussian_overlap(w,starts,max_sweeps=160,gradient_tolerance=1e-11)
        overlap=abs(contacts.conj()@w)
        order=np.argsort(overlap)[::-1]
        contacts=contacts[order]
        overlap=overlap[order]
        features,feature_report=full_distribution(contacts,model.j_matrices)
        geometry=[]
        arrays={'z':z,'witness':w,'contacts':contacts,'overlaps':overlap,'features':features['features']}
        for j in range(min(16,len(contacts))):
            report,raw=contact_hessian(w,contacts[j])
            geometry.append(report)
            arrays.update({f'contact{j}_{key}':value for key,value in raw.items()})
        np.savez_compressed(args.output/f't{t:g}_fixed.npz',**arrays)
        row={'t':t,'support_status':support_status,'local_support_max':float(overlap[0]),
             'support_upper_certified':1.,'support_value_exact':1. if t==0 else None,
             'fixed_before_replica_scan':True,'source_columns':str(args.columns.resolve()),
             'normalization_certificate_q4':certificate if t else None,
             'contact_geometry':geometry,'contact_features':feature_report,'search':diagnostic,
             'global_support_value_known':t==0,'numerical_contact_kets_exactly_certified':False,
             'elapsed_seconds':time.monotonic()-started}
        rows.append(row)
        write_json(args.output/'summary.json',{'status':'running','rows':rows})
        print(t,'h local',overlap[0],'zero Hessian',geometry[0]['zero_hessian_count_1e_6'],flush=True)
    write_json(args.output/'summary.json',{'status':'complete','rows':rows})


if __name__=='__main__':
    main()
