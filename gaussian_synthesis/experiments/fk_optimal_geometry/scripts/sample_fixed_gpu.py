#!/usr/bin/env python3
"""Fixed-direction C1/C2 scan using the read-only, validated CUDA SMC backend.

Independent runs, not individual correlated particles, are the statistical units.
Candidate support references are explicitly separated from global certificates.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import logsumexp
from fk5.paths import ROOT, PREVIOUS
from fk5.io import source_manifest, write_json
from fk4.cartan_sparse import CartanMoments
from fk4.distributions import full_distribution
from fk4.gpu_tempering import sample_tempered_gpu
from fk4.reference import log_dimension
from fk.model import build_model


def freeze(output, names):
    model = build_model()
    old = PREVIOUS/'results/fixed_geometry_01_qualified'
    old_rows = json.loads((old/'summary.json').read_text())['rows']
    cert = json.loads((ROOT/'results/initial_certificates_02/summary.json').read_text())
    specs = []
    for name in names:
        if name == 'toy_balanced':
            source = ROOT/'fk5/two_contact_toy.py'
            w = np.zeros(128,complex);w[0]=w[-1]=1
            h,exact,t=1.,True,0
            status='Exact vacuum/full two-contact toy; support=1 by the BCS product inequality.'
        elif name == 't20_joint':
            source = ROOT/'results/initial_certificates_02/t20_witness.npz'
            w = np.load(source)['joint_witness_display']
            row = next(r for r in cert['A1_rows'] if r['t'] == 20)
            h = row['support_upper']
            exact = True
            status = 'Analytic global envelope attained by A(2/(lambda-2), n); see joint contact theorem.'
            t = 20
        elif name == 't20_two_orbit':
            source = ROOT/'results/two_orbit_02/arc_dual_contacts.npz'
            w = np.load(source)['witness']
            h, exact, t = 1., False, 20
            status = 'Two-orbit matching candidate; h>=1 and searches found no excess. No global upper certificate.'
        else:
            t = int(name[1:])
            source = old/f't{t}_fixed.npz'
            w = np.load(source)['witness']
            row = next(r for r in old_rows if r['t'] == t)
            h = row['support_value_exact'] or row['local_support_max']
            exact = t == 0
            status = row['support_status']
        z = model.S.conj().T@w
        if name != 'toy_balanced':
            assert np.linalg.norm(model.S@z-w) < 1e-12
        np.savez_compressed(output/f'{name}_frozen.npz', witness=w, z=z)
        anchors = {1.: float(np.log(np.vdot(w, w).real)-log_dimension(8, 1))}
        for k in ([] if name == 'toy_balanced' else [2, 3, 4]):
            directory = PREVIOUS/f'results/cartan_exact_k{k}_01'
            assert json.loads((directory/'verification.json').read_text())['artifact_verification_passed']
            backend = CartanMoments(k, np.load(directory/'moment_matrix.npz')['matrix'])
            anchors[float(k)] = float(np.log(backend.evaluate(z, hessian=False).q)-log_dimension(8, k))
        specs.append(dict(name=name, t=t, reference_support=float(h), support_is_exact=exact,
                          support_status=status, anchors=anchors, source=str(source),
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    write_json(output/'frozen_witnesses.json', specs)
    return specs


def device_worker(device, jobs, config):
    model = build_model()
    output = Path(config['output'])
    completed = []
    for spec, seed in jobs:
        orders = spec.get('orders', config['orders'])
        name = f"{spec['name']}_seed{seed}"
        folder = output/name
        folder.mkdir(exist_ok=False)
        data = np.load(output/f"{spec['name']}_frozen.npz")
        w, z = data['witness'], data['z']
        observations = []
        anchors = {float(k): v for k, v in spec['anchors'].items()}
        def snapshot(kets, beta, log_z, log_r, diagnostic):
            if beta not in orders:
                return
            features, quality = full_distribution(kets, model.j_matrices)
            quality['distribution_kind'] = 'SMC approximation to P_k of a frozen direction; not the FK canonical witness w_k'
            f = features['features']
            deficit = 2*np.log(spec['reference_support'])-log_r
            log_weights = -.5*log_r
            weights = np.exp(log_weights-logsumexp(log_weights))
            row = dict(k=int(beta), log_q=float(log_dimension(8, beta)+log_z),
                       log_q_minus_2k_logh=float(log_dimension(8, beta)+log_z-2*beta*np.log(spec['reference_support'])),
                       overlap_log_deficit_mean=float(deficit.mean()),
                       overlap_width_rms=float(np.sqrt(max(0., deficit.mean()))),
                       particles_exceeding_reference=int(np.sum(deficit < -1e-8)),
                       P_features=quality, Lambda_feature_means=(weights@f).tolist(),
                       heuristic_eta_gt_08_P_mass=float(np.mean(f[:, 3] > .8)),
                       heuristic_eta_gt_08_Lambda_mass=float(weights@(f[:, 3] > .8)),
                       component_classifier_is_not_certified=True,
                       moment_anchor_used=beta in anchors)
            observations.append(row)
            extra = {}
            if spec['name'].startswith('toy_'):
                probabilities=abs(kets[:,[0,-1]])**2
                component=probabilities[:,1]>probabilities[:,0]
                row['full_contact_P_mass']=float(component.mean())
                row['full_contact_Lambda_mass']=float(weights@component)
                extra['vacuum_full_probabilities']=probabilities
            np.savez_compressed(folder/f'k{int(beta)}.npz',
                                sample_kets=kets[:512], log_r=log_r,
                                features=f, purity_errors=features['purity_errors'], **extra)
            write_json(folder/'progress.json', dict(observations=observations, last_stage=diagnostic))
            print(name, 'k', int(beta), 'logq-2klogh', row['log_q_minus_2k_logh'], flush=True)
        def stage(row):
            write_json(folder/'stage.json', row)
        try:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 810, spec['t'],
                                      int(hashlib.sha256(spec['name'].encode()).hexdigest()[:8], 16)]))
            result = sample_tempered_gpu(w, max(orders), config['count'], rng,
                                        sweeps=config['sweeps'], checkpoints=orders,
                                        state_callback=snapshot, callback=stage,
                                        log_z_anchors=anchors, device=device)
            report = dict(status='complete', witness=spec, seed=seed,
                          observations=observations, sampling=result.diagnostics)
            write_json(folder/'summary.json', report)
            completed.append(dict(name=name, device=device, elapsed_seconds=result.diagnostics['elapsed_seconds']))
        except Exception:
            write_json(folder/'failure.json', dict(traceback=traceback.format_exc()))
            raise
    return completed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=65536)
    p.add_argument('--sweeps', type=int, default=2)
    p.add_argument('--devices', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--seeds', type=int, nargs='+', default=[51001, 51002, 51003, 51004, 51005])
    p.add_argument('--orders', type=int, nargs='+', default=[8, 12, 16, 24, 32, 48, 64, 96, 128])
    p.add_argument('--names', nargs='+', default=['t0', 't20_joint', 't11', 't9', 't20_two_orbit'])
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output/'config.json', config)
    write_json(args.output/'manifest.json', source_manifest(args.output/'source_snapshot'))
    specs = freeze(args.output, args.names)
    jobs = [(spec, seed) for seed in args.seeds for spec in specs]
    started = time.monotonic()
    completed = []
    write_json(args.output/'summary.json', dict(status='running', total_jobs=len(jobs), completed=completed))
    with ProcessPoolExecutor(max_workers=len(args.devices), mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(device_worker, device, jobs[i::len(args.devices)], config)
                   for i, device in enumerate(args.devices)]
        for future in as_completed(futures):
            completed.extend(future.result())
            write_json(args.output/'summary.json', dict(status='running', total_jobs=len(jobs), completed=completed))
    write_json(args.output/'summary.json', dict(status='complete', total_jobs=len(jobs), completed=completed,
               elapsed_seconds=time.monotonic()-started, strictly_certified=False,
               uncertainty_unit='Independent seeded SMC runs, not resampled particles.'))


if __name__ == '__main__':
    main()
