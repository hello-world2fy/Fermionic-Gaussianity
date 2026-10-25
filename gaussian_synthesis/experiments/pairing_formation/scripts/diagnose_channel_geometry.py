#!/usr/bin/env python3
"""Post-screen orbital-overlap diagnostics, retaining the exact Pauli metric."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
import argparse
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model,ChannelObservables,linear_pair_matrix
from pairing_formation.linear import coupled_cluster_state
from pairing_formation.structure import collective_basis


def overlaps(model, forms, state, m):
    subsets,indices=model.blocks[m]
    block=state[indices]
    probability=float(np.linalg.norm(block)**2)
    if probability<1e-13:
        return {'m':m,'probability':probability}
    def evaluate(orbital):
        v=np.prod(orbital[np.array(subsets)],axis=1)
        v/=np.linalg.norm(v)
        # The hole profile is conjugately matched to the particle profile.
        joint=abs(np.vdot(np.outer(v,v.conj()),block))**2/probability
        upper=float(np.linalg.norm(v.conj()@block)**2/probability)
        return float(joint),upper
    angles=np.linspace(0,np.pi,181,endpoint=False)
    real=np.array([evaluate(forms@np.array([np.cos(b),np.sin(b)])) for b in angles])
    chiral=np.array([evaluate(forms@np.array([1.,sign*1j])/np.sqrt(2)) for sign in (-1,1)])
    best=int(np.argmax(real[:,0]))
    return {'m':m,'probability':probability,'best_real_angle':float(angles[best]),
            'best_real_joint_conditional_overlap':float(real[:,0].max()),
            'best_real_upper_conditional_overlap':float(real[:,1].max()),
            'chiral_joint_conditional_overlaps':chiral[:,0].tolist(),
            'chiral_upper_conditional_overlaps':chiral[:,1].tolist(),
            'joint_chiral_minus_best_real':float(chiral[:,0].max()-real[:,0].max())}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]/'results/pairing_channels_screen_01'
    records=[]
    for L in (12,16):
        for ratio in (0.,.85,1.):
            model,forms=angular_model(L,4.,ratio)
            data=np.load(root/f'L{L}_ratio{ratio:g}.npz')
            probe=np.array([.5,1.,2.,3.,4.])
            linear=linear_pair_matrix(model,probe)
            for t,T in zip(probe,linear):
                state=data['states'][np.argmin(abs(data['times']-t))]
                rpa=coupled_cluster_state(model,T)
                for name,vector in [('exact',state),('linear_lift',rpa)]:
                    records.append({'L':L,'ratio':ratio,'time':float(t),'source':name,
                                    'sectors':[overlaps(model,forms,vector,m) for m in (2,3)]})
    model,forms=angular_model(12)
    state=collective_basis(model,forms[:,0],forms[:,0])[:,2]
    counterexample={'scope':'Single real orbital per band at fixed transfer number m=2; maximal negative normalized fourth pair moment does not establish two chiral branches.',
                    'observation':ChannelObservables(model,forms).evaluate(state),
                    'geometry':overlaps(model,forms,state,2),
                    'single_pair_overlap':float(forms[:,0]@forms[:,1]),
                    'two_pair_overlap_unnormalized':float(2*((forms[:,0]@forms[:,1])**2-np.sum(forms[:,0]**2*forms[:,1]**2)))}
    (args.output/'evaluation.json').write_text(json.dumps({'scope':'Post-hoc geometry diagnostic, not a prediction or canonical-measure analysis. Overlaps use physical normalized orbital-product states, not bosonic-mode labels.',
                                                         'records':records,'counterexample':counterexample},indent=2))
    print(json.dumps(counterexample),flush=True)
