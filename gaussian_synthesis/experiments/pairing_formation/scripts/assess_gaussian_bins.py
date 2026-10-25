#!/usr/bin/env python3
"""Independent-replica stability of full-Gaussian coherent-bin quadrature.

This is a training-only diagnostic. Gaussian-bin integrals are coherent sums,
not individual Gaussian states. Block errors are empirical Monte Carlo errors,
not rigorous probability or interval certificates.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2 import Replica2


def column_basis(A):
    u,s,_=np.linalg.svd(A,full_matrices=False)
    return u[:,s>1e-11*s[0]]


def se(array):
    return np.sqrt(np.sum(abs(array-array.mean(axis=0))**2,axis=0)/(len(array)*(len(array)-1)))


def stability(blocks,base):
    # Shared known vectors are removed before judging additional structure.
    residual=blocks-(blocks@base.conj())@base.T
    # Exact coherent totals lie in base. This is the fixed 1/bin_count
    # control variate; the empirical bin-dependent noise remains.
    residual-=residual.mean(axis=1,keepdims=True)
    a=residual[::2].mean(axis=0);b=residual[1::2].mean(axis=0)
    mean=residual.mean(axis=0);sigma=se(residual)
    U,s,_=np.linalg.svd(mean.T,full_matrices=False)
    Ua,sa,_=np.linalg.svd(a.T,full_matrices=False)
    Ub,sb,_=np.linalg.svd(b.T,full_matrices=False)
    norm=float(np.linalg.norm(mean));noise=float(np.linalg.norm(sigma))
    rankchecks=[]
    for q in (1,2,4,8):
        if q>=min(mean.shape):continue
        cosine=np.linalg.svd(Ua[:,:q].conj().T@Ub[:,:q],compute_uv=False)
        rankchecks.append({'rank':q,'principal_cosines':cosine.tolist(),
          'smallest_singular_value_over_Frobenius_SE':float(s[q-1]/noise),
          'passes_declared_gate':bool(min(cosine)>=.95 and s[q-1]>=3*noise)})
    cross=(a@b.conj().T+b@a.conj().T)/2
    return {'additional_signal_norm':norm,'block_Frobenius_standard_error':noise,
      'relative_standard_error':noise/norm,'half_difference_over_half_sum':float(np.linalg.norm(a-b)/np.linalg.norm(a+b)),
      'singular_values':s.tolist(),'independent_cross_Gram_eigenvalues':np.linalg.eigvalsh(cross)[::-1].tolist(),
      'rank_checks':rankchecks},mean,U


def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    protocol={'classification':'Training-only independent Gaussian-bin quadrature assessment; no future state access.',
      'split':'Even-numbered GPU streams versus odd-numbered GPU streams, interleaved blocks for equal-size replicas.',
      'common_removed_directions':'F, exact training psi, and deterministic C2(psi), shared across all methods.',
      'fixed_control_variate':'Subtract equal share of sampled-total error from every bin; exact total is deterministic. No claim that this removes bin-dependent noise.',
      'stability_gate':'For ranks 1,2,4,8: every independent-half principal cosine >=0.95 and smallest pooled singular value >=3 times the Frobenius standard error of the pooled residual-bin matrix.',
      'restriction':'Passing stability alone is not a physical or predictive advantage. Full symmetry space has dimension 26. All errors here are empirical float64 Monte Carlo diagnostics.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    data=np.load(source/'training.npz');Q=data['Q'];model,_=angular_model(10)
    ref={'canonical':Q.T@data['target'],'linear_frame':Q.T@data['target'],'husimi_moment':Q.T@data['nonlinear_reference']}
    base=column_basis(Q.T@np.column_stack((model.initial,data['target'],data['nonlinear_reference'])))
    synth=Replica2(model).evaluate(data['witness'])[1]
    source_protocol=json.loads((source/'protocol.json').read_text());original=Path(source_protocol['source'])
    assert hashlib.sha256(original.read_bytes()).hexdigest()==source_protocol['source_sha256']
    streams=[np.load(source/f'device{d}.npz') for d in range(8)]
    metadata=[json.loads((source/f'device{d}.json').read_text()) for d in range(8)]
    assert len({row['samples'] for row in metadata})==1
    N=sum(row['samples'] for row in metadata)
    audit={'samples':N,'GPUs':8,'symmetry_dimension':Q.shape[1],'shared_removed_dimension':base.shape[1],
      'source_hash_verified':True,'deterministic_canonical_synthesis_residual':float(np.linalg.norm(synth-data['target'])),
      'symmetry_H_closure_error':float(np.linalg.norm(model.H@Q-Q@(Q.T@(model.H@Q)))),
      'training_symmetry_error':float(np.linalg.norm(data['target']-Q@(Q.T@data['target']))),'methods':{}}
    saved={'base':base,'Q':Q}
    for i,method in enumerate(('canonical','husimi_moment','linear_frame')):
        # Stream ordering keeps alternating rows in disjoint GPU pools.
        even=np.concatenate([streams[d][method+'_blocks'] for d in (0,2,4,6)])
        odd=np.concatenate([streams[d][method+'_blocks'] for d in (1,3,5,7)])
        blocks=np.stack((even,odd),axis=1).reshape(-1,12,Q.shape[1])
        total=blocks.sum(axis=1);raw=total.mean(axis=0);sigma=se(total)
        mass=np.mean([s['masses'][i] for s in streams],axis=0)
        w2=np.mean([s['weight_square_means'][i] for s in streams],axis=0)
        weighted_features=np.mean([s['weighted_coordinate_sums'][i] for s in streams],axis=0)
        residual_error=float(np.linalg.norm(raw-ref[method]));norm=float(np.linalg.norm(ref[method]))
        row={'raw_total_relative_error':residual_error/norm,
          'raw_total_relative_Frobenius_SE':float(np.linalg.norm(sigma)/norm),
          'raw_total_error_over_Frobenius_SE':float(residual_error/np.linalg.norm(sigma)),
          'total_coefficient_mass':float(mass.sum()),'coefficient_mass_fractions':(mass/mass.sum()).tolist(),
          'coefficient_weight_ESS':float(N*mass.sum()**2/w2.sum()),
          'bin_weight_ESS':(N*mass**2/w2).tolist(),
          'coefficient_weighted_configuration_coordinates':(weighted_features/mass[:,None]).tolist(),
          'raw_bin_vectors_standard_errors':np.linalg.norm(se(blocks),axis=1).tolist()}
        for label,grouped in (('count_and_circulation',blocks),('count_only',blocks.reshape(len(blocks),4,3,-1).sum(axis=2))):
            result,mean,U=stability(grouped,base);row[label]=result
            saved[method+'_'+label+'_residual']=mean
            saved[method+'_'+label+'_basis']=U
        corrected=blocks.mean(axis=0)+(ref[method]-raw)[None,:]/12
        row['corrected_total_error']=float(np.linalg.norm(corrected.sum(axis=0)-ref[method]))
        # Coherent Gram kernel: summing every entry recovers squared total norm.
        gram=corrected.conj()@corrected.T
        row['coherent_Gram_sum']=float(np.sum(gram).real)
        row['Gram_diagonal_sum']=float(np.trace(gram).real)
        saved[method+'_raw_bins']=blocks.mean(axis=0)
        saved[method+'_corrected_bins']=corrected
        saved[method+'_Gram']=gram
        audit['methods'][method]=row
    np.savez_compressed(out/'coherent_bins.npz',**saved)
    audit['any_resolved_rank_passes_gate']=any(x['passes_declared_gate'] for method in audit['methods'].values()
      for group in ('count_and_circulation','count_only') for x in method[group]['rank_checks'])
    audit['canonical_unique_physical_prediction_established']=False
    (out/'assessment.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.source,args.output)
