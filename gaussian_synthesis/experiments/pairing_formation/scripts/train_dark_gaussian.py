#!/usr/bin/env python3
"""Full-Gaussian k2 training at L14 before dark-conditioned coherent integrals."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy.sparse import load_npz
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2 import Replica2
from pairing_formation.replica2_gpu import Replica2GPU
from pairing_formation.dark_pairs import PairDarkSectors


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    rng = np.random.default_rng(943)
    checks = []
    for L in (4, 6, 8):
        model, _ = angular_model(L)
        cpu, gpu = Replica2(model), Replica2GPU(model)
        u = rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        u /= np.linalg.norm(u)
        v = rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        v /= np.linalg.norm(v)
        q0, C0 = cpu.evaluate(u); q1, C1 = gpu.evaluate(u)
        derivative = gpu.derivative(u, v)
        derivative_error = float(np.linalg.norm(derivative-cpu.derivative(u,v)))
        step = 1e-5
        fd = (gpu.evaluate(u+step*v)[1]-gpu.evaluate(u-step*v)[1])/(2*step)
        finite_difference_error = float(np.linalg.norm(fd-derivative))
        assert abs(q0-q1) < 1e-12 and np.linalg.norm(C0-C1) < 1e-12
        assert derivative_error < 1e-12 and finite_difference_error < 2e-10
        checks.append({'L': L, 'q_error': abs(q0-q1), 'C_error': float(np.linalg.norm(C0-C1)),
                       'derivative_error': derivative_error, 'finite_difference_error': finite_difference_error})
        del gpu
    (out/'backend_checks.json').write_text(json.dumps(checks, indent=2))
    model, forms = angular_model(14)
    Q = load_npz(out.parent/'rotation_geometry_01/L14_symmetry.npz').toarray()
    source = out.parent/'nonlinear_channels_01/L14_eta1_exact.npz'
    data = np.load(source); index = int(np.argmin(abs(data['times']-1.)))
    assert abs(data['times'][index]-1.) < 1e-12
    target = data['states'][index]
    replica = Replica2GPU(model)
    witness, report = replica.solve(target, Q)
    nonlinear_reference = replica.evaluate(target)[1]
    dark = PairDarkSectors(model, forms)
    np.savez_compressed(out/'training.npz', target=target, witness=witness, Q=Q,
                        nonlinear_reference=nonlinear_reference,
                        **{f'dark_m{m}': D for m,D in enumerate(dark.projectors)})
    report['seconds'] = time.monotonic()-begin
    report['direct_witness_ray_fidelity'] = float(abs(np.vdot(target,witness))**2/np.linalg.norm(witness)**2)
    (out/'synthesis.json').write_text(json.dumps(report, indent=2))
    (out/'protocol.json').write_text(json.dumps({
        'classification': 'Retrospective L14 G4 eta1 t1 training. Full-Gaussian k2 witness; no canonical sampling yet.',
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'training_sha256': hashlib.sha256((out/'training.npz').read_bytes()).hexdigest(),
        'candidate': 'Physical operator-defined dark populations in m>=2. Direct and unoptimized moment controls required.',
        'interpretation': 'Dark subspace is not a new channel by definition. It must show stable coherent measure information and useful predictions beyond Hamiltonian-generated response.'}, indent=2))
    print(json.dumps({k:report[k] for k in ('seconds','residual','iterations','direct_witness_ray_fidelity')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.output)
