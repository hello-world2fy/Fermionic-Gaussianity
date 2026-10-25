#!/usr/bin/env python3
"""Configuration-conditioned, phase-preserving full-Gaussian integrals.

Bins are calibrated once on an independent Haar pool. Independent GPU chains
are quadrature replicas, not iid samples from the canonical measure itself.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib,time
from math import comb
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.chirality_spectrum import ChiralitySpectrum
from pairing_formation.gaussian_bins import angular_invariant_basis,haar_atoms,atom_coordinates
from pairing_formation.projected_gaussian import torch_module
from pairing_formation.replica2 import Replica2


def setup(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent
    source=root/'angular_canonical_pilot_01/t1.npz';saved=np.load(source)
    model,_=angular_model(10);Q,_=angular_invariant_basis(model);probe=ChiralitySpectrum(model)
    K=np.zeros((model.L,model.L),complex);K[:model.r,:model.r]=probe.kernel;K[model.r:,model.r:]=probe.kernel
    chi=model.pair_operator(K).toarray();replica=Replica2(model)
    nonlinear_reference=replica.evaluate(saved['target'])[1]
    np.savez_compressed(out/'training.npz',target=saved['target'],witness=saved['witness'],Q=Q,chi=chi,nonlinear_reference=nonlinear_reference)
    protocol={'classification':'Training-only full-Gaussian coherent-integral pilot, L10 G4 eta1 t1. No independent new-parameter forecasts yet.',
      'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'dictionary':'Full even pure Gaussian Haar SO(4L), full-Fock normalization; target symmetry orbit averages used only for exact variance reduction.',
      'bin_coordinates':'Mean transferred pairs/r and absolute physical circulating-pair difference/r in the fixed-number paired projection of each Gaussian atom. These are configuration labels, not physical expectation values of the coherent target.',
      'calibration':'Independent 4096 Haar atoms; count quartiles x absolute-circulation terciles, shared by every method.',
      'integrands':{'canonical':'d2 |<g|u>|^2 <g|u> g','husimi_moment':'d2 |<g|psi>|^2 <g|psi> g; total is C2(psi), not psi','linear_frame':'d1 <g|psi> g'},
      'symmetry_dimension':Q.shape[1],'bin_count':12,
      'stability_gate':'Assess independent half-pool component errors and subspace principal cosines before any prediction. Noisy directions cannot count as discovered physics.',
      'physical_gate':'No important chiral cat has been established; compare the measured circulation spectrum against conventional nonlinear and generated-space controls.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch=torch_module();torch.set_num_threads(1);device='cuda:0';torch.manual_seed(841)
    counts=torch.as_tensor(model.m,dtype=torch.float64,device=device);ch=torch.as_tensor(chi,device=device)
    coordinates=[]
    for _ in range(8):coordinates.append(atom_coordinates(haar_atoms(model,512,device,torch),counts,ch,model.r,torch).cpu().numpy())
    coordinates=np.concatenate(coordinates)
    count_edges=np.r_[-1.,np.quantile(coordinates[:,0],[.25,.5,.75]),2.]
    chi_edges=np.r_[-1.,np.quantile(coordinates[:,1],[1/3,2/3]),2.]
    np.savez_compressed(out/'calibration.npz',coordinates=coordinates,count_edges=count_edges,chi_edges=chi_edges)
    (out/'calibration.json').write_text(json.dumps({'count_edges':count_edges.tolist(),'circulation_edges':chi_edges.tolist()},indent=2))
    print(json.dumps({'symmetry_dimension':Q.shape[1],'count_edges':count_edges.tolist(),'circulation_edges':chi_edges.tolist()}),flush=True)


def job(task):
    device,count,out=task;start=time.monotonic();torch=torch_module();torch.set_num_threads(1);torch.cuda.set_device(device)
    torch.manual_seed(842+device);dev=f'cuda:{device}'
    model,_=angular_model(10);data=np.load(out/'training.npz');calibration=np.load(out/'calibration.npz')
    target=torch.as_tensor(data['target'],device=dev);witness=torch.as_tensor(data['witness'],device=dev)
    Q=torch.as_tensor(data['Q'],device=dev,dtype=torch.complex128);chi=torch.as_tensor(data['chi'],device=dev)
    counts=torch.as_tensor(model.m,device=dev,dtype=torch.float64)
    ce=torch.as_tensor(calibration['count_edges'][1:-1],device=dev);he=torch.as_tensor(calibration['chi_edges'][1:-1],device=dev)
    d1=2**(2*model.L-1);d2=.5*comb(4*model.L,2*model.L)
    methods=('canonical','husimi_moment','linear_frame');blocks={name:[] for name in methods}
    masses=torch.zeros((3,12),device=dev,dtype=torch.float64);second_weights=torch.zeros_like(masses)
    features=torch.zeros((3,12,2),device=dev,dtype=torch.float64)
    block={name:torch.zeros((12,Q.shape[1]),device=dev,dtype=torch.complex128) for name in methods}
    diagnostics=[];last=time.monotonic();block_count=0
    for offset in range(0,count,512):
        batch=min(512,count-offset);atoms=haar_atoms(model,batch,dev,torch)
        coordinates=atom_coordinates(atoms,counts,chi,model.r,torch)
        labels=torch.bucketize(coordinates[:,0].contiguous(),ce)*3+torch.bucketize(coordinates[:,1].contiguous(),he)
        a=atoms.conj()@witness;b=atoms.conj()@target;projected=atoms@Q
        coefficients=(d2*a.abs()**2*a,d2*b.abs()**2*b,d1*b)
        weights=(d2*a.abs()**3,d2*b.abs()**3,d1*b.abs())
        for i,name in enumerate(methods):
            block[name].index_add_(0,labels,coefficients[i][:,None]*projected)
            masses[i].index_add_(0,labels,weights[i]);second_weights[i].index_add_(0,labels,weights[i]**2)
            features[i].index_add_(0,labels,weights[i][:,None]*coordinates)
        block_count+=batch
        if block_count>=4096 or offset+batch==count:
            for name in methods:blocks[name].append((block[name]/block_count).cpu().numpy());block[name].zero_()
            block_count=0
        if time.monotonic()-last>20:
            print(json.dumps({'device':device,'samples':offset+batch,'seconds':time.monotonic()-start}),flush=True);last=time.monotonic()
    np.savez_compressed(out/f'device{device}.npz',**{name+'_blocks':np.array(blocks[name]) for name in methods},
      masses=masses.cpu().numpy()/count,weight_square_means=second_weights.cpu().numpy()/count,
      weighted_coordinate_sums=features.cpu().numpy()/count)
    report={'device':device,'seed':842+device,'samples':count,'block_size':4096,'seconds':time.monotonic()-start}
    (out/f'device{device}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['setup','integrate'],required=True)
    p.add_argument('--per-device',type=int,default=131072);args=p.parse_args()
    if args.stage=='setup':setup(args.output)
    else:
        assert args.per_device%4096==0
        results=[]
        with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
            for f in as_completed([pool.submit(job,(d,args.per_device,args.output)) for d in range(8)]):results.append(f.result())
        (args.output/'integration_summary.json').write_text(json.dumps(results,indent=2))
