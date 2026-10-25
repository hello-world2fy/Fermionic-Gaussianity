#!/usr/bin/env python3
"""Independent-stream stability of actual dark-conditioned coherent moments."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,hashlib,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.dark_pairs import PairDarkSectors
from pairing_formation.rotation_geometry import orthogonal_columns


def se(blocks):
    return np.sqrt(np.sum(abs(blocks-blocks.mean(axis=0))**2,axis=0)/(len(blocks)*(len(blocks)-1)))


def assess(blocks,base):
    residual=blocks-(blocks@base.conj())@base.T
    pooled=residual.mean(axis=0)
    left=residual[::2].mean(axis=0);right=residual[1::2].mean(axis=0)
    U,s,_=np.linalg.svd(pooled.T,full_matrices=False)
    A,_,_=np.linalg.svd(left.T,full_matrices=False);B,_,_=np.linalg.svd(right.T,full_matrices=False)
    noise=float(np.linalg.norm(se(residual)))
    ranks=[]
    for q in range(1,len(s)+1):
        cosines=np.linalg.svd(A[:,:q].conj().T@B[:,:q],compute_uv=False)
        ranks.append({'rank':q,'principal_cosines':cosines.tolist(),
                      'smallest_singular_over_Frobenius_SE':float(s[q-1]/noise),
                      'passes_gate':bool(cosines[-1]>=.95 and s[q-1]>=3*noise)})
    return {'singular_values':s.tolist(),'Frobenius_SE':noise,
            'signal_norm':float(np.linalg.norm(pooled)),
            'half_difference_norm':float(np.linalg.norm(left-right)),
            'rank_checks':ranks,'largest_passing_rank':max([x['rank'] for x in ranks if x['passes_gate']]+[0])},U,pooled


def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    data=np.load(source/'training.npz');cal=np.load(source/'calibration.npz')
    protocol=json.loads((source/'protocol.json').read_text())
    assert hashlib.sha256(Path(protocol['source']).read_bytes()).hexdigest()==protocol['source_sha256']
    model,forms=angular_model(14);Q=data['Q'];psi=Q.T@data['target']
    q2=np.load(source.parent/'rotation_geometry_01/L14_frozen_spaces.npz')['source_q2']
    base,_=orthogonal_columns(np.column_stack((q2,psi,Q.T@data['nonlinear_reference'])),1e-11)
    dark=PairDarkSectors(model,forms)
    Dpsi=dark.apply(data['target'],'either');Dpsi[model.m<2]=0
    Hpsi=model.H@data['target'];DHpsi=dark.apply(Hpsi,'either');DHpsi[model.m<2]=0
    physical=np.column_stack((model.m*data['target'],model.m**2*data['target'],Dpsi,
                              model.m*Dpsi,model.m**2*Dpsi,model.H@Dpsi,DHpsi))
    physical=Q.T@physical;physical-=base@(base.conj().T@physical)
    physical_basis,_=orthogonal_columns(physical,1e-10)
    streams=[np.load(source/f'device{d}.npz') for d in range(8)]
    metadata=[json.loads((source/f'device{d}.json').read_text()) for d in range(8)]
    assert len({row['samples'] for row in metadata})==1
    inverse=np.linalg.inv(cal['whitening'].T)
    covariance=inverse@inverse.T
    groups={'joint':None,'count_only':[0,2,5],'dark_only':[1,4,8],
            'linear_features':[0,1],'linear_count':[0],'linear_dark':[1]}
    results={};saved={'base':base,'Q':Q,'physical_candidates':physical,'physical_basis':physical_basis}
    references={'canonical':psi,'linear_frame':psi,'husimi_moment':Q.T@data['nonlinear_reference']}
    for i,method in enumerate(('canonical','husimi_moment','linear_frame')):
        even=np.concatenate([streams[d][method+'_blocks'] for d in (0,2,4,6)])
        odd=np.concatenate([streams[d][method+'_blocks'] for d in (1,3,5,7)])
        blocks=np.stack((even,odd),axis=1).reshape(-1,10,Q.shape[1])
        total=blocks[:,0];error=float(np.linalg.norm(total.mean(axis=0)-references[method]))
        sigma=float(np.linalg.norm(se(total)));reference_norm=float(np.linalg.norm(references[method]))
        mass=float(np.mean([x['masses'][i] for x in streams]));second=float(np.mean([x['weight_square_means'][i] for x in streams]))
        row={'raw_total_relative_error':error/reference_norm,'raw_total_relative_Frobenius_SE':sigma/reference_norm,
             'raw_total_error_over_Frobenius_SE':error/sigma,'coefficient_mass':mass,
             'coefficient_ESS':sum(x['samples'] for x in metadata)*mass**2/second,'groups':{}}
        for group,indices in groups.items():
            moments=blocks[:,1:]
            if indices is not None:
                raw=np.einsum('ab,nbc->nac',inverse,moments)[:,indices]
                cov=covariance[np.ix_(indices,indices)]
                e,V=np.linalg.eigh(cov);whitening=(V/np.sqrt(e))@V.T
                moments=np.einsum('ab,nbc->nac',whitening.T,raw)
            result,U,mean=assess(moments,base)
            result['squared_projection_on_ordinary_physical_span']=np.sum(abs(physical_basis.conj().T@U)**2,axis=0).tolist()
            row['groups'][group]=result
            saved[f'{method}_{group}_basis']=U;saved[f'{method}_{group}_residual']=mean
        results[method]=row
    for method,row in results.items():
        for group,record in row['groups'].items():
            q=min(record['largest_passing_rank'],results['linear_frame']['groups'][group]['largest_passing_rank'])
            record['comparison_rank_with_linear']=q
            record['cosines_with_linear']=np.linalg.svd(saved[f'{method}_{group}_basis'][:,:q].conj().T@
                saved[f'linear_frame_{group}_basis'][:,:q],compute_uv=False).tolist() if q else []
    np.savez_compressed(out/'coherent_moments.npz',**saved)
    audit={'classification':'Training-only stability check; no future-state selection or fresh physical prediction.',
        'source':str(source),'samples':sum(x['samples'] for x in metadata),'GPUs':8,
        'shared_dimension':base.shape[1],'symmetry_dimension':Q.shape[1],
        'known_physical_comparison_dimension':physical_basis.shape[1],
        'methods':results,'source_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source/'training.npz',source/'calibration.npz',*[source/f'device{d}.npz' for d in range(8)]]},
        'moments_sha256':hashlib.sha256((out/'coherent_moments.npz').read_bytes()).hexdigest(),
        'rigorous_MC_confidence_certificate':False,'canonical_unique_physics_established':False}
    (out/'assessment.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'samples':audit['samples'],'shared_dimension':base.shape[1],
        'results':{method:{group:{k:record[k] for k in ('signal_norm','Frobenius_SE','largest_passing_rank','cosines_with_linear')}
            for group,record in row['groups'].items()} for method,row in results.items()}},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.source,args.output)
