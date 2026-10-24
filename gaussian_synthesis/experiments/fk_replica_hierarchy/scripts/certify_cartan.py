#!/usr/bin/env python3
"""Freeze canonical low-order witnesses from exact integer moment matrices."""
from pathlib import Path
import argparse
import sys
import time
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.cartan_sparse import CartanMoments
from fk4.exact_cartan import ExactCartan
from fk4.solver import solve
from fk4.io import source_manifest, write_json
from fk.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--k',type=int,required=True)
    p.add_argument('--exact',type=Path,required=True)
    p.add_argument('--previous',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    exact=ExactCartan(args.exact/'proof.json')
    previous=ExactCartan(args.previous/'proof.json') if args.previous else None
    backend=CartanMoments(args.k,np.load(args.exact/'moment_matrix.npz')['matrix'])
    model=build_model()
    rows=[]
    started=time.monotonic()
    for t in [0,5,9,9.1,10,10.9225,11,12,20]:
        target=model.target(t)
        target_z=model.S.conj().T@target
        solution=solve(backend,target_z)
        if not solution['converged']:
            raise ArithmeticError('canonical optimization failed')
        z=solution['z']
        certificate=exact.certificate(z,target,previous=previous)
        safe=certificate.pop('z_feasible')
        key=f't{t:g}_k{args.k}'
        np.savez_compressed(args.output/f'{key}_frozen.npz',z=z,witness=model.S@z,target=target,
                            target_z=target_z,z_feasible=safe)
        certificate.update(t=t,k=args.k,optimizer_gradient_norm=solution['gradient_norm'])
        rows.append(certificate)
        print(t,'L',certificate['L_certified_saved_target'],'rho upper',certificate['full_reconstruction_residual_upper'],flush=True)
        write_json(args.output/'summary.json',{'status':'running','rows':rows})
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'elapsed_seconds':time.monotonic()-started,
               'all_reconstruction_pass':all(r['full_reconstruction_residual_upper']<1e-3 for r in rows),
               'mass_integral_measured':False,'overall_E0_complete':False})


if __name__=='__main__':
    main()
