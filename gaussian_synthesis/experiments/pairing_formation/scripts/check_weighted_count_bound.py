#!/usr/bin/env python3
"""Retrospective fixed-number covariance certificates and independent checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.pairing_channels import angular_model
from pairing_formation.projected_count_bound import CanonicalAGPCheck
from pairing_formation.orbital_fluctuations import reduced_moments
from pairing_formation.weighted_count_bound import covariance_excess,weighted_statistics,weighted_fidelity_bound,weighted_echo


def free_checks():
    rng=np.random.default_rng(823);checks=[];rotated=[]
    for L in range(2,7):
        for M in range(1,L+1):
            check=CanonicalAGPCheck(L,M);max_excess=-np.inf;max_cov=-np.inf
            for index in range(40):
                log=(1 if index%2 else 5)*rng.normal(size=L)+1j*rng.normal(size=L)
                c=check.coefficients(log);p=abs(c)**2;n=p@check.occupations
                C=check.occupations.T@(p[:,None]*check.occupations)-np.outer(n,n)
                max_cov=max(max_cov,float(np.linalg.eigvalsh(covariance_excess(n,C,M))[-1]))
                raw=rng.normal(size=(2*L,2*L))+1j*rng.normal(size=(2*L,2*L))
                K=(raw+raw.conj().T)/np.linalg.norm(raw+raw.conj().T)
                if index%10==0:K=np.eye(2*L)
                action=np.zeros(len(check.configurations),complex)
                np.add.at(action,check.dst,check.signs*K[check.ci,check.cj]*c[check.src])
                mean=float(np.vdot(c,action[check.paired_indices]).real)
                variance=float(np.linalg.norm(action)**2-mean**2)
                bound=float(2*np.repeat(n,2)@np.diag(K@K).real-mean**2/M)
                max_excess=max(max_excess,variance-bound)
            assert max_excess<3e-11 and max_cov<3e-12
            checks.append({'L':L,'M':M,'cases':40,'max_full_Fock_excess':max_excess,'max_covariance_eigenvalue':max_cov})
    for L,M in ((3,1),(3,2),(4,2),(5,2),(5,3)):
        check=CanonicalAGPCheck(L,M);excesses=[]
        for _ in range(20):
            U,_=np.linalg.qr(rng.normal(size=(2*L,2*L))+1j*rng.normal(size=(2*L,2*L)))
            state=check.rotated_full_state(rng.normal(size=L)+1j*rng.normal(size=L),U)
            weights=rng.normal(size=2*L);A=weights[check.configurations].sum(axis=1)
            K2=(weights**2)[check.configurations].sum(axis=1);p=abs(state)**2
            mean=p@A;variance=p@(A*A)-mean**2;bound=2*(p@K2)-mean**2/M
            excesses.append(float(variance-bound))
        assert max(excesses)<2e-11
        rotated.append({'L':L,'M':M,'cases':20,'max_full_Pfaffian_excess':max(excesses)})
    return {'arbitrary_Hermitian_onebody':checks,'independent_rotated_Pfaffian':rotated}


def main(output):
    output.mkdir(parents=True,exist_ok=False);root=output.parent
    protocol={'classification':'Retrospective observable selection and analytic-family exclusion, not a new-parameter forecast or canonical-specific mechanism.',
      'theorem':'Var(c^dagger K c) <= 2 Tr(gamma K^2) - Tr(gamma K)^2/M, N=2M, K Hermitian, pure total-number-projected Gaussian.',
      'scope':'No extra seniority, spin, or spatial many-body projection. The ordinary variance bound is not valid for classical mixtures; the QFI extension is separate.',
      'optimality':'Largest covariance eigenvalue optimizes this energy-diagonal quadratic witness at Euclidean unit weight norm; it does not optimize Gaussian fidelity.',
      'interval_certified':False,'novelty_priority_established':False}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    checks=free_checks();(output/'independent_checks.json').write_text(json.dumps(checks,indent=2))
    cases=[]
    for t in (.5,1.,2.15):
        cases.append((f'uniform_L12_t{t:g}',picket_fence(12,2.),t,np.load(root/f'replica2_L12_G2_01/t{t:g}.npz')['target']))
    for L,eta in ((10,1.),(14,1.),(18,1.),(14,.85)):
        model,_=angular_model(L,4.,relative_strength=eta)
        cases.append((f'angular_L{L}_eta{eta:g}_t2',model,2.,np.load(root/f'nonlinear_channels_01/L{L}_eta{eta:g}_exact.npz')['states'][40]))
    rows=[];early_weight=None;pulse_weight=None
    for name,model,t,state in cases:
        moments=reduced_moments(model,state);matrix=covariance_excess(moments['occupations'],moments['covariance'],model.r)
        values,vectors=np.linalg.eigh(matrix);weights=vectors[:,-1]
        if weights[-1]<0:weights=-weights
        ordinary=weighted_statistics(model,state,np.r_[np.zeros(model.r),np.ones(model.r)])
        result=weighted_fidelity_bound(model,state,weights) if model.L<=14 else weighted_statistics(model,state,weights)
        angles=np.array([.001,.003,.01,.02,.03,.05,.1,.2,.3,.5])
        echo=weighted_echo(model,state,weights,angles)
        row={'name':name,'L':model.L,'time':t,'largest_eigenvalue':float(values[-1]),'weights':weights.tolist(),
             'eigenpair_residual':float(np.linalg.norm(matrix@weights-values[-1]*weights)),
             'upper_band_count':ordinary,'weighted':result,
             'echo':{key:value.tolist() for key,value in echo.items()}}
        if name=='uniform_L12_t1':early_weight=weights
        if name=='uniform_L12_t2.15':
            pulse_weight=weights
            e,U=eigh(model.H.toarray());independent=U@(np.exp(-1j*t*e)*(U.conj().T@model.initial))
            overlap=np.vdot(independent,state);independent*=overlap/abs(overlap)
            row['independent_dense_eigh']={'state_difference':float(np.linalg.norm(independent-state)),
              'frozen_weight_excess':weighted_statistics(model,independent,weights)['excess']}
            row['early_selected_weight_excess']=weighted_statistics(model,state,early_weight)['excess']
        np.savez_compressed(output/f'{name}.npz',state=state,weights=weights,eigenvalues=values,excess_matrix=matrix,**moments)
        rows.append(row)
        print(json.dumps({'name':name,'largest_eigenvalue':row['largest_eigenvalue'],'weighted':result}),flush=True)
    pilot=json.loads((root/'cluster_feedback_pilot_01/evaluation.json').read_text())
    comparisons=[];model=picket_fence(12,2.)
    for entry in pilot['models']:
        state=np.load(root/'cluster_feedback_pilot_01'/f"{entry['name']}_prediction.npz")['states'][43]
        comparisons.append({'name':entry['name'],**weighted_statistics(model,state,pulse_weight)})
    (output/'evaluation.json').write_text(json.dumps({'cases':rows,'cluster_pilot_fixed_pulse_weight':comparisons},indent=2))
    pulse_state=next(state for name,_,_,state in cases if name=='uniform_L12_t2.15')
    rounded=[]
    for decimals in (1,2,3,4):
        w=np.round(pulse_weight,decimals)
        rounded.append({'decimal_places':decimals,'weights':w.tolist(),**weighted_statistics(model,pulse_state,w)})
    (output/'rounded_weights.json').write_text(json.dumps(rounded,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
