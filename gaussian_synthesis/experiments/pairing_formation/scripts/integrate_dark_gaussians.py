#!/usr/bin/env python3
"""Full Gaussian coherent moments of count and multiple-transfer darkness."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse, hashlib, json, time, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import comb
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.gaussian_bins import haar_atoms
from pairing_formation.projected_gaussian import torch_module


def coordinates(atoms, model, training, torch):
    norm = (atoms.abs()**2).sum(dim=1)
    counts = torch.as_tensor(model.m, device=atoms.device, dtype=torch.float64)
    mean = (atoms.abs()**2) @ counts / norm / model.r
    probability = torch.zeros_like(norm)
    for m in range(2, model.r+1):
        P = training[f'dark_m{m}']
        if np.linalg.norm(P) < 1e-12: continue
        D = torch.as_tensor(P, device=atoms.device)
        indices = torch.as_tensor(model.blocks[m][1], device=atoms.device)
        C = atoms[:, indices]
        left, right = D @ C, C @ D.T
        both = left @ D.T
        probability += (left.abs()**2 + right.abs()**2 - both.abs()**2).sum(dim=(1,2))
    return torch.stack((mean, probability/norm), dim=1)


def monomials(x, torch):
    c, d = x[:,0], x[:,1]
    return torch.stack((c,d,c*c,c*d,d*d,c**3,c*c*d,c*d*d,d**3), dim=1)


def setup(out, source, reuse=None):
    out.mkdir(parents=True, exist_ok=False)
    if reuse is not None:
        import shutil
        for name in ('training.npz','calibration.npz'):
            shutil.copyfile(reuse/name,out/name)
        protocol=json.loads((reuse/'protocol.json').read_text())
        protocol['reused_calibration_from']=str(reuse)
        protocol['followup']='Independent quadrature after a noise-dominated 262144-atom pilot. Original feature definitions fixed; scalar and degree-one feature subsets are additional retrospective stability diagnostics. No forecasts selected from the first pool.'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        print(json.dumps({'stage':'setup','reused_calibration':str(reuse)}),flush=True)
        return
    training = np.load(source/'training.npz')
    np.savez_compressed(out/'training.npz', **{k:training[k] for k in training.files})
    model, _ = angular_model(14)
    torch = torch_module(); torch.set_num_threads(1); torch.manual_seed(951)
    values = []
    begin = time.monotonic()
    for _ in range(16):
        atoms = haar_atoms(model,256,'cuda:0',torch)
        values.append(coordinates(atoms,model,training,torch).cpu().numpy())
    values = np.concatenate(values)
    mean, scale = values.mean(axis=0), values.std(axis=0)
    raw = monomials(torch.as_tensor((values-mean)/scale),torch).numpy()
    center = raw.mean(axis=0)
    covariance = (raw-center).T @ (raw-center) / len(raw)
    eigenvalues, U = np.linalg.eigh(covariance)
    assert eigenvalues[0] > 1e-8
    whitening = (U / np.sqrt(eigenvalues)) @ U.T
    np.savez_compressed(out/'calibration.npz', coordinates=values,mean=mean,scale=scale,
                        feature_mean=center,whitening=whitening,covariance_eigenvalues=eigenvalues)
    protocol = {'classification':'Previously inspected L14 G4 eta1 t1; actual full-Gaussian weighted coherent integrals, not iid canonical measure draws.',
        'source':str(source/'training.npz'),'source_sha256':hashlib.sha256((source/'training.npz').read_bytes()).hexdigest(),
        'dictionary':'Full SO(56) even Gaussian Haar, full-Fock normalized before extracting fixed-N seniority-zero amplitudes.',
        'coordinates':['Mean transferred pairs/r in projected atom','Either-band source-annihilator darkness restricted to m>=2 in projected atom'],
        'features':'All nonconstant two-coordinate monomials of total degree <=3, standardized and whitened using 4096 independent calibration atoms. Labels are not target physical observables.',
        'integrands':{'canonical':'d2 |<g|u>|^2 <g|u> f(g) g','husimi_moment':'d2 |<g|psi>|^2 <g|psi> f(g) g','linear_frame':'d1 <g|psi> f(g) g'},
        'symmetry_average':'Exact invariant Q of dimension170; scalar features invariant under same cyclic/reflection/band-exchange group.',
        'stability_gate':'Independent GPU-half principal cosines >=.95 and last retained singular value >=3 times Frobenius standard error before prediction.',
        'strong_controls':'Same common source and training span, ordinary physical-operator moments, linear frame, unoptimized cubic, Hamiltonian response.',
        'calibration_seconds':time.monotonic()-begin}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'stage':'setup','seconds':protocol['calibration_seconds'],'mean':mean.tolist(),'scale':scale.tolist()}),flush=True)


def job(task):
    device, count, batch_size, seed, out = task
    begin=time.monotonic(); torch=torch_module();torch.set_num_threads(1)
    torch.cuda.set_device(device);torch.manual_seed(seed+device);dev=f'cuda:{device}'
    model,_=angular_model(14);training=np.load(out/'training.npz');cal=np.load(out/'calibration.npz')
    Q=torch.as_tensor(training['Q'],device=dev,dtype=torch.complex128)
    target=torch.as_tensor(training['target'],device=dev);witness=torch.as_tensor(training['witness'],device=dev)
    mean=torch.as_tensor(cal['mean'],device=dev);scale=torch.as_tensor(cal['scale'],device=dev)
    center=torch.as_tensor(cal['feature_mean'],device=dev);whitening=torch.as_tensor(cal['whitening'],device=dev)
    d1=2**(2*model.L-1);d2=.5*comb(4*model.L,2*model.L)
    methods=('canonical','husimi_moment','linear_frame')
    blocks={name:[] for name in methods}
    current={name:torch.zeros((10,Q.shape[1]),device=dev,dtype=torch.complex128) for name in methods}
    masses=torch.zeros(3,device=dev,dtype=torch.float64)
    second=torch.zeros_like(masses);block_count=0;last=time.monotonic()
    for offset in range(0,count,batch_size):
        batch=min(batch_size,count-offset)
        atoms=haar_atoms(model,batch,dev,torch)
        x=coordinates(atoms,model,training,torch)
        features=(monomials((x-mean)/scale,torch)-center)@whitening
        features=torch.cat((torch.ones((batch,1),device=dev),features),dim=1).to(torch.complex128)
        a=atoms.conj()@witness;b=atoms.conj()@target;projected=atoms@Q
        coefficients=(d2*a.abs()**2*a,d2*b.abs()**2*b,d1*b)
        weights=(d2*a.abs()**3,d2*b.abs()**3,d1*b.abs())
        for i,name in enumerate(methods):
            current[name] += features.T@(coefficients[i][:,None]*projected)
            masses[i] += weights[i].sum();second[i] += (weights[i]**2).sum()
        block_count+=batch
        if block_count>=4096 or offset+batch==count:
            for name in methods:
                blocks[name].append((current[name]/block_count).cpu().numpy())
                current[name].zero_()
            block_count=0
        if time.monotonic()-last>20:
            print(json.dumps({'device':device,'samples':offset+batch,'seconds':time.monotonic()-begin}),flush=True);last=time.monotonic()
    np.savez_compressed(out/f'device{device}.npz',**{name+'_blocks':np.array(blocks[name]) for name in methods},
                        masses=masses.cpu().numpy()/count,weight_square_means=second.cpu().numpy()/count)
    report={'device':device,'seed':seed+device,'samples':count,'batch_size':batch_size,'block_size':4096,'seconds':time.monotonic()-begin}
    (out/f'device{device}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['setup','integrate'],required=True)
    p.add_argument('--source',type=Path,default=Path('results/dark_gaussian_training_01'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--per-device',type=int,default=32768)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--seed',type=int,default=952)
    p.add_argument('--reuse-calibration',type=Path);args=p.parse_args()
    if args.stage=='setup':setup(args.output,args.source,args.reuse_calibration)
    else:
        assert args.per_device%4096==0 and 4096%args.batch_size==0
        assert not list(args.output.glob('device*')),'Do not overwrite existing quadrature output'
        reports=[]
        with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
            for f in as_completed([pool.submit(job,(d,args.per_device,args.batch_size,args.seed,args.output)) for d in range(8)]):reports.append(f.result())
        (args.output/'integration_summary.json').write_text(json.dumps(reports,indent=2))
