#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.multipair_noise import transfer_operators,noise_observables


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent;model,_=angular_model(14);operators=transfer_operators(model)
    exact=np.load(root/'nonlinear_channels_01/L14_eta1_exact.npz');references=[noise_observables(operators,v) for v in exact['states']]
    rows=[];hashes={}
    for method in ('exact','full_cc','source_only_cc','riccati'):
        path=root/f'nonlinear_channels_01/L14_eta1_{method}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();data=np.load(path)
        for k,(t,v) in enumerate(zip(data['times'],data['states'])):
            v=v/np.linalg.norm(v);means,coherence,variance=noise_observables(operators,v);ref=references[k]
            rows.append({'method':method,'time':float(t),'two_transfer_connected_L2_error':float(np.linalg.norm(coherence-ref[1])),
                'pair_pulse_variance_L2_error':float(np.linalg.norm(variance-ref[2])),
                'exact_connected_noise_norm':float(np.linalg.norm(ref[1])),
                'connected_real':coherence.real.tolist(),'connected_imag':coherence.imag.tolist(),'pulse_variances':variance.tolist(),
                'variance_anisotropy_identity_error':float(np.max(abs(variance[::2]-variance[1::2]-4*np.diag(coherence).real)))})
    # A short Hermitian pair-transfer pulse has survival loss epsilon^2 Var(Q).
    v=exact['states'][40];_,_,variance=noise_observables(operators,v);pulse_checks=[]
    for q in range(4):
        for quadrature in (0,1):
            A=operators[q];Q=A+A.conj().T if quadrature==0 else -1j*(A-A.conj().T)
            estimates=[]
            for epsilon in (.002,.001):
                moved=expm_multiply(-1j*epsilon*Q,v,traceA=0.)
                estimates.append(float((1-abs(np.vdot(v,moved))**2)/epsilon**2))
            richardson=(4*estimates[1]-estimates[0])/3
            pulse_checks.append({'q':q,'quadrature':quadrature,'variance':float(variance[2*q+quadrature]),'survival_loss_estimates':estimates,'Richardson_error':float(abs(richardson-variance[2*q+quadrature]))})
    report={'classification':'Retrospective physical observable screen on existing trajectories; no canonical discovery or new holdouts.','input_hashes':hashes,'rows':rows,'pair_pulse_checks':pulse_checks}
    (out/'screen.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'primary':[r for r in rows if abs(r['time']-2)<1e-12],
        'max_variance_identity_error':max(r['variance_anisotropy_identity_error'] for r in rows),
        'max_pulse_Richardson_error':max(r['Richardson_error'] for r in pulse_checks)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
