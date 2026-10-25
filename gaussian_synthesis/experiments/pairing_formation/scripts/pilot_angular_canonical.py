#!/usr/bin/env python3
"""Bounded full-Gaussian witness pilot; never label it a new forecast."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2 import Replica2
from pairing_formation.frozen import channel_basis


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['synthesize','analyze'],required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]/'results'
    model,_=angular_model(10)
    V=np.exp(2j*np.pi*np.arange(model.r)[:,None]*np.arange(model.r)[None,:]/model.r)/np.sqrt(model.r)
    if args.stage=='synthesize':
        args.output.mkdir(parents=True,exist_ok=False)
        protocol={'scope':'Posthoc bounded training-geometry pilot after inspecting baseline late spectra; not a new prediction.',
          'L':10,'G':4.,'eta':1.,'training_times':[.25,.5,1.],
          'question':'Does the k2 full-Gaussian witness prioritize the uniform angular channel earlier than the exact m1 amplitude? Compare direct, variational CC, projective CC and Hamiltonian-derived Pauli channels before any predictive claim.',
          'measure_scope':'Witness coordinates are not samples of the canonical measure.'}
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        replica=Replica2(model);data=np.load(root/'nonlinear_channels_01/L10_eta1_exact.npz')
        results=[]
        def spectrum(x):
            C=V.conj().T@x[model.blocks[1][1]]@V
            return {'probabilities':(abs(np.diag(C))**2/np.linalg.norm(C)**2).tolist(),
                    'offdiagonal_relative_norm':float(np.linalg.norm(C-np.diag(np.diag(C)))/np.linalg.norm(C))}
        for t in (.25,.5,1.):
            s=data['states'][int(round(t/.05))];start=time.monotonic();u,report=replica.solve(s)
            np.savez_compressed(args.output/f't{t:g}.npz',target=s,witness=u,reconstruction=replica.evaluate(u)[1])
            report.update({'time':t,'state_spectrum':spectrum(s),'witness_spectrum':spectrum(u),'seconds':time.monotonic()-start})
            for method,folder in [('full_cc','nonlinear_channels_01'),('projective','projective_channels_01')]:
                states=np.load(root/f'{folder}/L10_eta1_{method}.npz')['states']
                report[method+'_spectrum']=spectrum(states[int(round(t/.05))])
            (args.output/f't{t:g}.json').write_text(json.dumps(report,indent=2));results.append(report)
        (args.output/'summary.json').write_text(json.dumps(results,indent=2))
    else:
        checks=[]
        # Keep entire degenerate pairs. At t=1 the uniform k=0 direction
        # is the third singular direction for both exact state and witness.
        for t,allowed in [(.25,(2,4)),(.5,(2,4)),(1.,(2,3))]:
            data=np.load(args.output/f't{t:g}.npz')
            matrices=[data[key][model.blocks[1][1]] for key in ('target','witness')]
            for q in allowed:
                A,_=channel_basis(model,matrices[0],q)
                B,_=channel_basis(model,matrices[1],q)
                error=float(np.linalg.norm(A@A.conj().T-B@B.conj().T))
                assert A.shape==B.shape
                checks.append({'time':t,'channels':q,'dimension':A.shape[1],
                               'full_generated_span_projector_difference':error})
        (args.output/'span_comparison.json').write_text(json.dumps({
          'scope':'Only m1-singular-vector generated Gaussian spans are compared. This is not a no-go theorem for all information in the full canonical measure.',
          'degeneracy_policy':'Never truncate inside a paired degenerate Fourier eigenspace.',
          'checks':checks,'max_projector_difference':max(x['full_generated_span_projector_difference'] for x in checks)},indent=2))
        print(json.dumps(checks,indent=2))
