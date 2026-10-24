#!/usr/bin/env python3
"""Check the integer Hessian and its full-Fock Gaussian chart independently."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4.io import write_json,source_manifest
from fk3.discovery import covariance_from_kets


def pf(A):
    if len(A)==0:return 1.+0j
    value=0j
    for j in range(1,len(A)):
        selected=[a for a in range(len(A)) if a not in (0,j)]
        value+=(-1)**(j+1)*A[0,j]*pf(A[np.ix_(selected,selected)])
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    pairs=[(i,j) for i in range(8) for j in range(i+1,8)];indices={x:j for j,x in enumerate(pairs)}
    H=np.eye(56,dtype=np.int64)
    for offset in [0,4]:
        for pair1,pair2,sign in [((0,1),(2,3),1),((0,2),(1,3),-1),((0,3),(1,2),1)]:
            a=indices[tuple(j+offset for j in pair1)];b=indices[tuple(j+offset for j in pair2)]
            H[2*a,2*b]=H[2*b,2*a]=-sign
            H[2*a+1,2*b+1]=H[2*b+1,2*a+1]=sign
    I=np.eye(56,dtype=np.int64)
    assert np.array_equal(H,H.T) and np.count_nonzero(H@(H-I)@(H-2*I))==0
    assert np.trace(H)==56 and np.trace(H@H)==80
    # Symmetry + the exact annihilating polynomial restrict eigenvalues to
    # 0,1,2; dimension and two traces then give multiplicities 12,32,12.
    masks=[m for m in range(256) if m.bit_count()%2==0]
    w=np.array([1. if m in [0,15,240,255] else 0. for m in masks],complex)
    rng=np.random.default_rng(34291);errors=[];kets=[]
    for _ in range(64):
        x=rng.normal(size=56);x/=np.linalg.norm(x);loss=[]
        for step in [.01,.005]:
            Z=np.zeros((8,8),complex)
            for j,(a,b) in enumerate(pairs):Z[a,b]=step*(x[2*j]+1j*x[2*j+1]);Z[b,a]=-Z[a,b]
            g=[]
            for mask in masks:
                occupied=[j for j in range(8) if mask>>j&1]
                g.append(pf(Z[np.ix_(occupied,occupied)]))
            g=np.array(g);g/=np.linalg.norm(g);kets.append(g)
            loss.append(-np.log(abs(np.vdot(w,g))))
        recovered=(16*loss[1]-loss[0])/(3*.01**2)
        errors.append(abs(recovered-.5*x@H@x))
    cov=covariance_from_kets(np.array(kets));purity=float(np.max(abs(cov@cov+np.eye(16))))
    assert max(errors)<5e-8 and purity<1e-10
    np.savez_compressed(args.output/'hessian.npz',H_integer=H,pairs=np.array(pairs),finite_chart_errors=errors)
    write_json(args.output/'verification.json',{'artifact_verification_passed':True,
        'exact_integer_annihilating_polynomial':'H(H-I)(H-2I)=0',
        'exact_dimension_trace_trace_square':[56,56,80],
        'exact_eigenvalue_multiplicities':{'0':12,'1':32,'2':12},
        'normal_dimension':44,'Laplace_exponent':22,'predicted_B_minus_2_leading_coefficient':22,
        'full_Fock_random_chart_directions_checked':64,'finite_chart_Hessian_error_max':max(errors),
        'Gaussian_chart_purity_error_max':purity,
        'finite_k_asymptotic_remainder_certified':False})
    print('exact Hessian multiplicities 12,32,12; full-Fock error',max(errors))


if __name__=='__main__':main()
