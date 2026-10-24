#!/usr/bin/env python3
"""Rebuild full Fock kernels, SVD variational energies and Wick contractions."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import write_json
from fk.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    args=p.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    config=json.loads((args.run/'config.json').read_text())
    if summary['status']!='complete':raise ValueError('run not complete')
    manifest=json.loads((args.run/'manifest.json').read_text())
    for rel,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((args.run/'source_snapshot'/rel).read_bytes()).hexdigest()==digest
    model=build_model()
    masks=model.even_masks
    perm=np.argmax(abs(model.gammas),axis=2)
    phases=np.take_along_axis(model.gammas,perm[:,:,None],axis=2)[:,:,0]
    def covariance(states):
        states=np.atleast_2d(states)
        full=np.zeros((len(states),256),complex);full[:,masks]=states
        acted=full[:,perm]*phases[None]
        cov=(1j*np.einsum('nai,nbi->nab',acted.conj(),acted,optimize=True)).real
        for a in range(16):cov[:,a,a]=0
        return .5*(cov-cov.transpose(0,2,1))
    terms=[]
    for J in model.j_matrices:
        pairs=[(a,b,J[a,b]) for a in range(8) for b in range(a+1,8) if J[a,b]!=0]
        terms.extend((a,b,c+8,d+8,.25*x*y) for a,b,x in pairs for c,d,y in pairs)
    def wick(cov):
        return sum(v*(cov[...,a,b]*cov[...,c,d]-cov[...,a,c]*cov[...,b,d]+cov[...,a,d]*cov[...,b,c]) for a,b,c,d,v in terms)
    operators={'mass_correlation':model.mass_correlation_operator[np.ix_(masks,masks)]}
    for indices in [(0,1,8,9),(0,2,8,10),(0,1,2,3)]:
        O=np.eye(256,dtype=complex)
        for j in indices:O=O@model.gammas[j]
        operators['fourpoint_'+'_'.join(map(str,indices))]=O[np.ix_(masks,masks)]
    targets={}
    for entry in summary['targets']:
        t,eps=entry['t'],entry['epsilon']
        full=model.h_int.copy()
        for a in range(8):full+=.5j*t*(1+eps*(1 if a<4 else -1))*(model.gammas[a]@model.gammas[a+8])
        H=full[np.ix_(masks,masks)]
        E=np.linalg.eigvalsh(full)[0]
        target=np.load(args.run/f't{t:g}_eps{eps:g}_target.npz')['target']
        assert np.linalg.norm(H@target-E*target)<1e-9
        assert abs(E-entry['energy'])<1e-9
        targets[t,eps]=(H,target,float(E))
    # Compute each family's maximum atom set once and use its prefixes.
    groups={}
    for r in summary['rows']:
        key=(r['t'],r['epsilon'],r['replicate'],r['method'])
        if key not in groups or r['budget']>groups[key]['budget']:groups[key]=r
    maxima={}
    errors={name:0. for name in ['kernel','state','energy','SVD_energy','fidelity','Gaussian_purity','Wick','mixture','parts','frozen_atoms','total_cost']}
    def check(name,value,tolerance=1e-8):
        value=float(np.max(abs(value)))
        errors[name]=max(errors[name],value)
        if value>tolerance:raise ArithmeticError(f'{name} verification error {value}')
    frozen=Path(config['frozen'])
    for key,r in groups.items():
        data=np.load(args.run/f"{r['key']}.npz")
        atoms=data['atoms']
        cov=covariance(atoms)
        check('Gaussian_purity',cov@cov+np.eye(16),1e-9)
        if 'frozen_source_key' in r:
            source=frozen/f"{r['frozen_source_key']}.npz"
            assert hashlib.sha256(source.read_bytes()).hexdigest()==r['frozen_source_sha256']
            check('frozen_atoms',atoms-np.load(source)['atoms'][:len(atoms)],1e-13)
        maxima[key]=(atoms,cov)
    started=time.monotonic()
    for i,r in enumerate(summary['rows']):
        data=np.load(args.run/f"{r['key']}.npz")
        atoms,c,state=data['atoms'],data['coefficients'],data['state']
        H,target,E=targets[r['t'],r['epsilon']]
        phi=atoms.T
        count=len(atoms)
        assert count==r['actual_gaussian_terms']==r['budget']
        if r['method']=='single_gaussian_HFB':assert count==1
        key=(r['t'],r['epsilon'],r['replicate'],r['method'])
        maxatoms,covs=maxima[key]
        check('frozen_atoms',atoms-maxatoms[:count],1e-13)
        covs=covs[:count]
        check('state',phi@c-state,1e-10)
        check('state',np.vdot(state,state)-1,1e-10)
        energy=float(np.vdot(state,H@state).real)
        check('energy',energy-r['energy'])
        check('energy',energy-E-r['energy_error'])
        fidelity=float(abs(np.vdot(target,state))**2)
        check('fidelity',fidelity-r['fidelity'])
        U,s,V=np.linalg.svd(phi,full_matrices=False)
        keep=s*s>r['rank_tolerance']*max(1.,float(s[0]**2))
        assert int(keep.sum())==r['retained_rank']
        Q=U[:,keep]
        variational=np.linalg.eigvalsh(Q.conj().T@H@Q)[0]
        check('SVD_energy',energy-variational,2e-7)
        weights=abs(c)**2;weights/=weights.sum()
        cov=covariance(state)[0]
        mixcov=np.einsum('i,ijk->jk',weights,covs)
        coherent_mass=float(np.vdot(state,operators['mass_correlation']@state).real)
        target_mass=float(np.vdot(target,operators['mass_correlation']@target).real)
        target_wick=target_mass-wick(covariance(target)[0])
        check('Wick',abs(coherent_mass-target_mass)-r['mass_correlation_error'])
        check('Wick',abs(coherent_mass-wick(cov)-target_wick)-r['wick_deviation_error'])
        check('Wick',coherent_mass-wick(cov)-r['coherent']['wick_deviation'])
        check('Wick',wick(cov)-r['coherent']['wick_prediction'])
        check('Wick',coherent_mass-r['coherent']['mass_correlation'])
        # Parts were explicitly divided by c* S c. Its rounding error is
        # amplified by large cancelling diagonal/interference contributions;
        # replacing it by the mathematical value one changes the reported
        # finite-precision definition for ill-conditioned dictionaries.
        denominator=float(np.vdot(c,(phi.conj().T@phi)@c).real)
        for name,O in {'overlap':np.eye(128),'hamiltonian':H,**operators}.items():
            K=phi.conj().T@O@phi
            check('kernel',K-data[name])
            total=float(np.vdot(state,O@state).real)
            diagonal=float(np.sum(abs(c)**2*np.diag(K)).real/denominator)
            part=r['coherent_kernel_parts'][name]
            check('parts',np.array([total-part['total'],diagonal-part['diagonal'],total-diagonal-part['off_diagonal']]))
            if name in ('hamiltonian','mass_correlation'):
                measure=float(weights@np.diag(K).real)
                field='energy' if name=='hamiltonian' else name
                check('mixture',measure-r['incoherent'][field])
        check('mixture',wick(mixcov)-r['incoherent']['wick_prediction'])
        check('mixture',r['incoherent']['mass_correlation']-wick(mixcov)-r['incoherent']['wick_deviation'])
        cost=sum(r[k] for k in ['source_generation_seconds_cold','dictionary_generation_seconds','dictionary_selection_seconds','prediction_seconds'])
        check('total_cost',cost-r['total_cold_prediction_seconds'],1e-10)
        assert r['test_target_used_for_candidates']==('test_oracle' in r['method'])
        if i%250==0:print('verified',i,'/',len(summary['rows']),flush=True)
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,
               'rows':len(summary['rows']),'maximum_errors':errors,
               'independent_checks':['full256 Hamiltonians and parity ground energies','SVD subspace eigenvalues','full256 Majorana covariance','explicit three-term Wick contractions','all coherent kernels and interference','Gaussian purity of every distinct saved atom','actual term counts','frozen source hashes and exact atoms','reported total costs','source snapshots'],
               'elapsed_seconds':time.monotonic()-started,
               'source_target_separation_note':'Source hashes and execution ordering are checked; this does not turn oracle methods into prediction methods.'})


if __name__=='__main__':main()
