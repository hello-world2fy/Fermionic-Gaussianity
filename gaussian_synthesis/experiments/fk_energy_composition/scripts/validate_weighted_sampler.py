#!/usr/bin/env python3
"""Independent CP1 quadrature regression for weighted Gaussian SMC."""
from pathlib import Path
import argparse
import os
import sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from numpy.polynomial.legendre import leggauss
from fk6.io import source_manifest,write_json
from fk6.weighted_sampler import sample_weighted
from fk4.gpu_tempering import sample_tempered_gpu


def reference(u,penalty,endpoint,order):
    x,w=leggauss(order);p=(x+1)/2;weights=w/2
    angle=2*np.pi*np.arange(2*order)/(2*order)
    g=np.stack(np.broadcast_arrays(np.sqrt(1-p)[:,None],np.sqrt(p)[:,None]*np.exp(1j*angle)),axis=-1)
    overlap=g.conj()@u;density=(abs(overlap)/(1+penalty*p[:,None]))**(2*endpoint)
    Z=float(weights@density.mean(axis=1))
    moments=np.array([weights@(density*p[:,None]).mean(axis=1),weights@(density*p[:,None]**2).mean(axis=1),
        weights@(density*(g[:,:,0].conj()*g[:,:,1]).real).mean(axis=1),
        weights@(density*(g[:,:,0].conj()*g[:,:,1]).imag).mean(axis=1)])/Z
    return Z,moments


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);write_json(out/'manifest.json',source_manifest(out/'source_snapshot'))
    u=np.array([.8,.3+.5196152422706632j]);u/=np.linalg.norm(u);H=np.diag([0.,1.])
    plain=sample_tempered_gpu(u,2.,256,np.random.default_rng(511),n=2,device=0,sweeps=3)
    zero=sample_weighted(u,H,0.,1.,0.,2.,256,np.random.default_rng(511),n=2,device=0)
    assert np.array_equal(plain.kets,zero.kets) and plain.log_z==zero.log_z
    rows=[]
    for penalty,endpoint in [(1.,1.5),(4.,3.5)]:
        Z,expected=reference(u,penalty,endpoint,128);Z2,expected2=reference(u,penalty,endpoint,256)
        assert max(abs(expected-expected2))<1e-7 and abs(Z-Z2)<1e-7
        observations=[]
        for seed in [25101,25102,25103,25104,25105]:
            sample=sample_weighted(u,H,0.,1.,penalty,endpoint,32768,np.random.default_rng(seed),n=2,device=0)
            g=sample.kets;p=abs(g[:,1])**2
            observed=np.array([p.mean(),(p*p).mean(),np.mean((g[:,0].conj()*g[:,1]).real),np.mean((g[:,0].conj()*g[:,1]).imag)])
            observations.append(dict(seed=seed,observed=observed,normalizer=np.exp(sample.log_z),sampler=sample.diagnostics))
            np.savez_compressed(out/f'lambda{penalty:g}_beta{endpoint:g}_seed{seed}.npz',kets=g)
        values=np.array([r['observed'] for r in observations]);error=abs(values.mean(axis=0)-expected2)
        tolerance=6*values.std(axis=0,ddof=1)/np.sqrt(5)+.0005
        passed=bool(np.all(error<tolerance))
        rows.append(dict(penalty=penalty,endpoint=endpoint,expected=expected2,normalizer=Z2,observations=observations,
            errors=error,tolerances=tolerance,passed=passed))
        print('lambda',penalty,'beta',endpoint,'error',error,'passed',passed,flush=True)
    write_json(out/'verification.json',dict(passed=all(r['passed'] for r in rows),zero_penalty_bitwise_regression=True,
        quadrature='Independent Gauss-Legendre p and periodic phase integration on CP1, 128 vs256 convergence',
        distribution_checks_are_statistical=True,rows=rows))
    if not all(r['passed'] for r in rows):raise ArithmeticError('Weighted sampler distribution regression failed')


if __name__=='__main__':main()
