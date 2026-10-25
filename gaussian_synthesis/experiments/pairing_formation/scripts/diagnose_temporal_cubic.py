#!/usr/bin/env python3
"""Post-prediction capacity and physical tangent diagnostics; not forecasts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.cubic_manifold import CubicManifold
from pairing_formation.pairing_channels import angular_model
from pairing_formation.dark_pairs import PairDarkSectors


def phase_distance(a,b):
    overlap=np.vdot(b,a)
    return float(np.linalg.norm(a*np.exp(-1j*np.angle(overlap))-b))


def fit(task):
    method,time,source,out=task;data=np.load(source/'frozen_bases.npz');T=np.load(source/'frozen_tensors.npz')[method]
    model=CubicManifold(T,data['H']);trajectory=np.load(source/f'{method}_restart.npz')
    index=int(round((time-1.35)*20));c0=trajectory['coordinates'][index]
    exact=np.load(source.parent/'nonlinear_channels_01/L14_eta1_exact.npz')['states'][round(time*20)]@data['Q'];k=len(c0)
    def objective(x):
        c=x[:k]+1j*x[k:];psi,J=model.tangent(c);a=np.vdot(exact,psi)
        return 1-abs(a)**2,-2*np.real(a.conjugate()*(exact.conj()@J))
    rng=np.random.default_rng(1550+round(time*20));starts=[c0,data[method].conj().T@data['endpoint_witness']]
    starts +=[c0+.3*np.linalg.norm(c0)*(rng.normal(size=k)+1j*rng.normal(size=k))/np.sqrt(2*k) for _ in range(4)]
    attempts=[];best=None
    for c in starts:
        result=minimize(objective,np.r_[c.real,c.imag],jac=True,method='L-BFGS-B',
            options={'maxiter':1200,'ftol':1e-14,'gtol':1e-10,'maxcor':30})
        candidate=result.x[:k]+1j*result.x[k:];value=objective(result.x)[0]
        attempts.append({'fidelity':float(1-value),'success':bool(result.success),'iterations':int(result.nit),
            'gradient_norm':float(np.linalg.norm(objective(result.x)[1]))})
        if best is None or value<best[0]:best=(value,candidate)
    psi,J=model.tangent(best[1]);dc=model.rhs(time,best[1]);velocity=J@np.r_[dc.real,dc.imag]
    Hp=data['H']@psi;truth=-1j*(Hp-np.vdot(psi,Hp)*psi)
    path=out/f'{method}_t{time:.2f}.npz';np.savez_compressed(path,coefficients=best[1],state=psi,target=exact,normal_rate=truth-velocity)
    report={'method':method,'time':time,'local_feasible_fidelity':float(1-best[0]),
        'actual_prediction_fidelity':float(abs(np.vdot(exact,trajectory['states'][index]))**2),
        'state_rate_residual_at_local_fit':float(np.linalg.norm(truth-velocity)),
        'attempts':attempts,'global_fidelity_bound':False,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps({k:v for k,v in report.items() if k!='attempts'}),flush=True);return report


def main(source,out):
    out.mkdir(parents=True,exist_ok=False);data=np.load(source/'frozen_bases.npz');T=np.load(source/'frozen_tensors.npz')
    physical,forms=angular_model(14);Q=data['Q'];counts=np.rint((Q*Q).T@physical.m).astype(int)
    dark=PairDarkSectors(physical,forms);D=Q.T@dark.apply(Q,'either');D*=((counts>=2)[:,None]&(counts>=2)[None,:])
    endpoint=[];refinements=[]
    for method in ('canonical','direct','Hamiltonian','common'):
        m=CubicManifold(T[method],data['H']);c=data[method].conj().T@data['endpoint_witness']
        psi,J=m.tangent(c);dc=m.rhs(1.35,c);v=J@np.r_[dc.real,dc.imag];Hp=data['H']@psi
        true=-1j*(Hp-np.vdot(psi,Hp)*psi);normal=true-v
        dp=np.array([2*np.vdot(psi[counts==j],v[counts==j]).real for j in range(8)])
        exact_dp=np.array([2*np.vdot(psi[counts==j],true[counts==j]).real for j in range(8)])
        endpoint.append({'method':method,'tangent_error':float(np.linalg.norm(normal)),
            'current_continuity_L1':float(np.sum(abs(dp-exact_dp))),
            'mean_transfer_rate_error':float(abs((dp-exact_dp)@np.arange(8))),
            'normal_dark_squared_fraction':float(np.vdot(normal,D@normal).real/np.linalg.norm(normal)**2),
            'normal_sector_squared_weights':[float(np.linalg.norm(normal[counts==j])**2) for j in range(8)]})
        for start in ('Fermi','restart'):
            old=np.load(source/f'{method}_{start}.npz');new=np.load(source/f'{method}_{start}_refined.npz')
            refinements.append({'method':method,'start':start,
                'max_phase_aligned_state_change':max(phase_distance(a,b) for a,b in zip(old['states'],new['states']))})
    (out/'protocol.json').write_text(json.dumps({'scope':'After frozen prediction evaluation, diagnose known-condition ansatz capacity and Hamiltonian tangent error. All target fits are retrospective; they must never be counted as predictions or global optima.',
        'times':[1.35,2.,2.5],'starts_per_fit':6,'endpoint_controls':endpoint,'numerical_refinements':refinements,
        'source_bases_sha256':hashlib.sha256((source/'frozen_bases.npz').read_bytes()).hexdigest(),
        'source_tensors_sha256':hashlib.sha256((source/'frozen_tensors.npz').read_bytes()).hexdigest()},indent=2))
    tasks=[(method,time,source,out) for method in ('canonical','direct','Hamiltonian') for time in (1.35,2.,2.5)];records=[]
    with ProcessPoolExecutor(max_workers=9) as pool:
        for f in as_completed([pool.submit(fit,t) for t in tasks]):records.append(f.result())
    (out/'fits.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.source,args.output)
