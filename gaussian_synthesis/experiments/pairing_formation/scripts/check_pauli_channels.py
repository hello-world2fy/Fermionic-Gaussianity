#!/usr/bin/env python3
"""Verify the short-time Pauli source and diagnose its later exact weight."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
import argparse
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model,linear_pair_matrix


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    checks=[]
    for L in (10,12,14,16):
        for G in (3.,4.):
            model,forms=angular_model(L,G,1.)
            U,_=np.linalg.qr(forms)
            P=U@U.T; Q=np.eye(model.r)-P
            vec=model.initial.copy()
            orders=[]
            for n in (1,2,3):
                vec=model.H@vec
                orders.append(Q@vec[model.blocks[1][1]]@Q)
            expected=-4*Q@(model.kernel[model.r:,:model.r]**3)@Q
            errors=[float(np.linalg.norm(orders[0])),float(np.linalg.norm(orders[1])),
                    float(np.linalg.norm(orders[2]-expected))]
            assert max(errors)<2e-10
            cubic=1j*expected/6
            times=np.array([.01,.02,.04,.08])
            states=model.evolve(times)
            probabilities=[]
            for t,state in zip(times,states):
                C=state[model.blocks[1][1]]
                probability=float(np.linalg.norm(Q@C@Q)**2)
                probabilities.append({'time':float(t),'dark_probability':probability,
                                      'leading_probability':float(np.linalg.norm(cubic)**2*t**6),
                                      'relative_probability_correction':float(probability/(np.linalg.norm(cubic)**2*t**6)-1)})
            linear=linear_pair_matrix(model,[.5,1.,2.])
            checks.append({'L':L,'G':G,'identity_errors':errors,
                           'max_Riccati_dark_norm':max(float(np.linalg.norm(Q@T@Q)) for T in linear),
                           'cubic_squared_norm':float(np.linalg.norm(cubic)**2),
                           'smallest_two_times_probability_power':float(np.log(probabilities[1]['dark_probability']/probabilities[0]['dark_probability'])/np.log(2)),
                           'short_times':probabilities})
    root=Path(__file__).resolve().parents[1]/'results/pairing_channels_screen_01'
    late=[]
    for L in (12,16):
        model,forms=angular_model(L)
        r=model.r
        V=np.exp(2j*np.pi*np.arange(r)[:,None]*np.arange(r)[None,:]/r)/np.sqrt(r)
        source=np.zeros(r,bool);source[[1,r-1]]=True
        data=np.load(root/f'L{L}_ratio1.npz')
        rows=[]
        for t,state in zip(data['times'],data['states']):
            C=state[model.blocks[1][1]]
            transformed=V.conj().T@C@V
            weights=abs(np.diag(transformed))**2
            probability=float(np.linalg.norm(C)**2)
            dark=float(weights[~source].sum())
            rows.append({'time':float(t),'m1_probability':probability,'fourier_schmidt_weights':weights.tolist(),
                         'dark_probability':dark,'dark_fraction_given_m1':dark/probability if probability>1e-14 else 0.,
                         'circulant_offdiagonal_error':float(np.linalg.norm(transformed-np.diag(np.diag(transformed))))})
        late.append({'L':L,'G':4.,'rows':rows})
    report={'scope':'Exact short-time mechanism and retrospective late weights; no priority or canonical-only discovery claim. The competing-channel extension still requires nonlinear pair-CC/SMF and predictive deletion controls.',
            'formula':'Q C(t) Q = -(2i/3) t^3 Q (J_ph elementwise-cubed) Q + O(t^4), for real equal-strength angular channels and flat shells.',
            'interpretation':'At fixed finite size, hard-core exclusion generates one-transfer Schmidt harmonics absent from the attractive interaction and the finite-size corrected Riccati solution. RPA failure alone is not enough to establish the research goal.',
            'checks':checks,'late':late}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'cases':len(checks),'maximum_identity_error':max(max(c['identity_errors']) for c in checks),
                      'Riccati_max_dark_norm':max(c['max_Riccati_dark_norm'] for c in checks)}),flush=True)
