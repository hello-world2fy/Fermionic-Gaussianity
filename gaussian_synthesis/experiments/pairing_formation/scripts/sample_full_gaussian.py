#!/usr/bin/env python3
"""Independent full-Haar pilot of the k=2 canonical coefficient distribution.

Canonical weights are |<g|u>|^3; Husimi weights are |<g|psi>|^2.
Raw atoms keep the full-Fock normalization. Saved weighted reservoirs are
for discovery, not certified sparse decompositions or independent iid draws.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from math import comb
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.projected_gaussian import torch_module, torch_pfaffian


def job(task):
    device, count, source, output = task
    started = time.monotonic()
    torch = torch_module()
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    dev = f'cuda:{device}'
    torch.manual_seed(20261101+device)
    model = picket_fence(12, 2.)
    n, dim = 2*model.L, model.dim
    data = np.load(source)
    target = torch.tensor(data['target'], dtype=torch.complex128, device=dev)
    witness = torch.tensor(data['witness'], dtype=torch.complex128, device=dev)
    idx = torch.tensor([[k for j in range(model.L) if mask & (1 << j)
                        for k in (2*j, 2*j+1)] for mask in model.masks], device=dev)
    reservoirs = {name: None for name in ('canonical', 'husimi', 'haar')}
    statistics = {name: np.zeros(3) for name in reservoirs}
    reconstruction = np.zeros(dim, complex)
    block_reconstructions = []
    moments = []
    for offset in range(0, count, 128):
        batch = min(128, count-offset)
        normal = torch.randn((batch, 2*n, 2*n), device=dev, dtype=torch.float64)
        O, R = torch.linalg.qr(normal)
        O = O*torch.where(torch.diagonal(R, dim1=1, dim2=2) >= 0, 1., -1.)[:, None, :]
        sign, _ = torch.linalg.slogdet(O)
        O[:, :, -1] *= sign[:, None]
        U = .5*(O[:, ::2, ::2]+O[:, 1::2, 1::2]+1j*(O[:, 1::2, ::2]-O[:, ::2, 1::2]))
        V = .5*(O[:, ::2, ::2]-O[:, 1::2, 1::2]+1j*(O[:, 1::2, ::2]+O[:, ::2, 1::2]))
        Z = -torch.linalg.solve(U, V)
        Z = .5*(Z-Z.transpose(1, 2))
        vacuum = torch.exp(.5*torch.linalg.slogdet(U)[1])
        atoms = torch_pfaffian(Z[:, idx[:, :, None], idx[:, None, :]], torch)*vacuum[:, None]
        a, b = atoms.conj()@witness, atoms.conj()@target
        dimension = .5*comb(2*n, n)
        contribution = dimension*((a.abs()**2*a)[:, None]*atoms)
        block_sum = contribution.sum(dim=0).cpu().numpy()
        reconstruction += block_sum
        block_reconstructions.append(block_sum/batch)
        moments.extend((dimension*a.abs()**4).cpu().numpy().tolist())
        sector_weight = (atoms.abs()**2).sum(dim=1)
        weights = {'canonical': a.abs()**3, 'husimi': b.abs()**2, 'haar': torch.ones_like(a.real)}
        for name, weight in weights.items():
            statistics[name] += np.array([float(weight.sum()), float((weight**2).sum()),
                                         float((weight*sector_weight).sum())])
            uniform = torch.rand(batch, device=dev, dtype=torch.float64).clamp_min(1e-300)
            key = weight.clamp_min(1e-300).log()-(-uniform.log()).log()
            candidate = (key, atoms, Z, vacuum)
            old = reservoirs[name]
            if old is not None:
                candidate = tuple(torch.cat((x, y), dim=0) for x, y in zip(old, candidate))
            selected = torch.topk(candidate[0], min(256, len(candidate[0]))).indices
            reservoirs[name] = tuple(x[selected] for x in candidate)
    reconstructed = reconstruction/count
    moments = np.array(moments)
    exact = data['reconstruction']
    report = {'device': device, 'seed': 20261101+device, 'count': count,
              'full_physical_modes': n, 'dictionary': 'SO(2n) full Haar Gaussian',
              'q_mc': float(moments.mean()), 'q_mc_se': float(moments.std(ddof=1)/np.sqrt(count)),
              'q_deterministic': float(np.vdot(data['witness'], exact).real),
              'sector_reconstruction_mc_error': float(np.linalg.norm(reconstructed-exact)),
              'B_mc': float(dimension*statistics['canonical'][0]/count),
              'statistics': {key: {'importance_ess': float(s[0]**2/s[1]),
                             'weighted_atom_sector_probability': float(s[2]/s[0])}
                             for key, s in statistics.items()},
              'certified': False, 'reservoir_sampling': 'Gumbel weighted sampling without replacement',
              'wall_seconds': time.monotonic()-started}
    for name, reservoir in reservoirs.items():
        arrays = [x.cpu().numpy() for x in reservoir]
        np.savez_compressed(output/f'device{device}_{name}.npz', keys=arrays[0], atoms=arrays[1],
                            Z=arrays[2], vacuum=arrays[3])
    np.savez_compressed(output/f'device{device}_moments.npz', reconstruction=reconstructed,
                        block_reconstructions=np.array(block_reconstructions))
    (output/f'device{device}.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--per-device', type=int, default=32768)
    p.add_argument('--devices', type=int, nargs='+', default=list(range(8)))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=len(args.devices), mp_context=multiprocessing.get_context('spawn')) as pool:
        for report in pool.map(job, [(d, args.per_device, args.source, args.output) for d in args.devices]):
            print(json.dumps(report), flush=True)
