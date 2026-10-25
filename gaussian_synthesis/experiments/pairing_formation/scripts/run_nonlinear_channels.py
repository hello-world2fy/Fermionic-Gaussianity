#!/usr/bin/env python3
"""Odd-grid nonlinear pairing controls, with forecasts saved before references."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model, linear_pair_matrix
from pairing_formation.pair_cc_dynamics import PairCCDynamics, product_parameter_map
from pairing_formation.linear import coupled_cluster_state
from pairing_formation.kernel_stochastic import torch_kernel_run

CASES = [(10, 1.), (14, 1.), (18, 1.), (14, .85)]
TIMES = np.linspace(0., 2.5, 51)


def name(L, eta):
    return f'L{L}_eta{eta:g}'


def projectors(r, forms):
    U, _ = np.linalg.qr(forms)
    a = 2*np.pi*np.arange(r)/r
    U3, _ = np.linalg.qr(np.column_stack((np.cos(3*a), np.sin(3*a))))
    return U, U@U.T, U3@U3.T


def cc_job(task):
    L, eta, output = task
    start = time.monotonic()
    model, forms = angular_model(L, 4., eta)
    U, _, _ = projectors(model.r, forms)
    reports = {}
    for method, M in [('full_cc', None), ('source_only_cc', product_parameter_map(U, U))]:
        dynamics = PairCCDynamics(model, M)
        states, coordinates, report = dynamics.evolve(TIMES)
        np.savez_compressed(output/f'{name(L,eta)}_{method}.npz', times=TIMES, states=states, coordinates=coordinates)
        reports[method] = report
        print(json.dumps({'condition':name(L,eta),'method':method,'seconds':time.monotonic()-start,
                          'norm_error':report['max_norm_error'],'energy_drift':report['max_energy_drift']}), flush=True)
    matrices = linear_pair_matrix(model, TIMES)
    states = np.array([coupled_cluster_state(model, T) for T in matrices])
    np.savez_compressed(output/f'{name(L,eta)}_riccati.npz', times=TIMES, states=states, matrices=matrices)
    (output/f'{name(L,eta)}_cc_diagnostics.json').write_text(json.dumps(reports, indent=2))
    return {'condition':name(L,eta),'seconds':time.monotonic()-start}


def stochastic_job(task):
    L, eta, kind, device, output, count = task
    model, forms = angular_model(L, 4., eta)
    _, P, P3 = projectors(model.r, forms)
    reports = []
    for seed, dt in [(2026,.004), (2027,.004), (2026,.002)]:
        report, blocks = torch_kernel_run(model, [P,P3], count=count, kind=kind, seed=seed,
                                           times=TIMES, dt=dt, device=device)
        tag = f'{name(L,eta)}_{kind}_seed{seed}_dt{dt:g}'
        np.savez_compressed(output/f'{tag}.npz', times=TIMES, blocks=blocks)
        (output/f'{tag}.json').write_text(json.dumps(report, indent=2))
        reports.append({'tag':tag, **{k:report[k] for k in ('max_spin_length_squared_drift','max_number_drift','max_energy_symbol_drift')}})
        print(json.dumps(reports[-1]), flush=True)
    return reports


def state_rows(model, states, forms):
    _, P, P3 = projectors(model.r, forms)
    Q = np.eye(model.r)-P
    ops = []
    for projector in (P,P3):
        W = np.zeros((model.L, model.L))
        W[model.r:,model.r:] = projector
        ops.append(model.pair_operator(W))
    rows = []
    for t, raw in zip(TIMES,states):
        norm = np.linalg.norm(raw)
        state = raw/norm
        count = np.bincount(model.m, weights=abs(state)**2, minlength=model.r+1)
        m = np.arange(model.r+1)
        mean = float(count@m); variance = float(count@(m*m)-mean*mean)
        C = state[model.blocks[1][1]]
        p1 = float(np.linalg.norm(C)**2)
        dark = float(np.linalg.norm(Q@C@Q)**2)
        echo = float(1-abs(count@np.exp(-.2j*m))**2-.04*mean)
        rows.append({'time':float(t),'count':count.tolist(),'mean':mean,'variance':variance,
                     'm1_probability':p1,'m1_dark_probability':dark,
                     'm1_dark_fraction':dark/p1 if p1>1e-14 else None,
                     'm1_outside_source_probability':float(p1-np.linalg.norm(P@C@P)**2),
                     'upper_source_occupation':float(np.vdot(state,ops[0]@state).real),
                     'upper_third_harmonic_occupation':float(np.vdot(state,ops[1]@state).real),
                     'echo_witness_theta02':echo,'norm_error':float(abs(norm-1)),
                     'energy':float(np.vdot(state,model.H@state).real)})
    return rows


def evaluation_job(task):
    L, eta, output = task
    model, forms = angular_model(L, 4., eta)
    states = model.evolve(TIMES)
    np.savez_compressed(output/f'{name(L,eta)}_exact.npz', times=TIMES, states=states)
    exact = state_rows(model, states, forms)
    predictions = {}
    for method in ('full_cc','source_only_cc','riccati'):
        predicted = np.load(output/f'{name(L,eta)}_{method}.npz')['states']
        rows = state_rows(model, predicted, forms)
        for row, truth, p, s in zip(rows,exact,predicted,states):
            row['fidelity'] = float(abs(np.vdot(p,s))**2/np.linalg.norm(p)**2)
            row['count_TV'] = float(.5*np.sum(abs(np.array(row['count'])-truth['count'])))
        predictions[method] = rows
    report = {'L':L,'G':4.,'eta':eta,'exact':exact,'predictions':predictions}
    (output/f'{name(L,eta)}_evaluation.json').write_text(json.dumps(report,indent=2))
    t2 = {'condition':name(L,eta),'time':2.,'exact':exact[40]}
    for method, rows in predictions.items():
        t2[method] = {k:rows[40][k] for k in ('fidelity','count_TV','m1_dark_probability','upper_third_harmonic_occupation')}
    print(json.dumps(t2), flush=True)
    return t2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['init','cc','stochastic','evaluate'],required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--samples',type=int,default=262144)
    args = parser.parse_args()
    if args.stage == 'init':
        args.output.mkdir(parents=True,exist_ok=False)
        protocol = {'scope':'Strong nonlinear controls for the existing finite angular-channel mechanism; not a canonical forecast or a new phase claim.',
                    'cases':CASES,'G':4.,'times':TIMES.tolist(),'primary_time':2.,
                    'prior_information':'Even grids and short-time odd-grid coefficients are known. A full-CC L10 eta1 pilot has been inspected before this protocol; no new-condition exact late states have been read.',
                    'same_initial_state':'All deterministic models start from the identical exact Fermi sea.',
                    'methods':['Hermitian Dirac-Frenkel full pair-CC','same TDVP with T confined to the two source orbitals','finite-size corrected Riccati lift','Gaussian-seeded spin mean field','discrete TWA'],
                    'stochastic_samples_per_run':args.samples,'seeds':[2026,2027],'dt':[.004,.002],
                    'stochastic_observables':['mean transferred pairs','operator-symbol count variance','upper pair occupation in source and third-angular-harmonic subspaces'],
                    'unavailable_stochastic_observables':['pure-state echo','postselected coherent m1 amplitude'],
                    'causal_test':'Deleting all non-source directions in the full nonlinear CC manifold while holding the initial state fixed.',
                    'failure_rule':'If conventional nonlinear CC generates the cubic new direction, that direction alone cannot establish canonical-specific discovery. A late deviation requires further physical interpretation and stronger alternatives.',
                    'reference_policy':'Save all deterministic and stochastic predictions, then hash them before any new exact reference calculation.'}
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    elif args.stage in ('cc','stochastic'):
        if args.stage == 'cc':
            tasks = [(L,e,args.output) for L,e in CASES]; function = cc_job
        else:
            tasks = [(L,e,k,d,args.output,args.samples) for d,(L,e,k) in enumerate((L,e,k) for L,e in CASES for k in ('gaussian','discrete'))]
            function = stochastic_job
        summaries = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(function,t) for t in tasks]):
                summaries.append(future.result())
        (args.output/f'{args.stage}_summary.json').write_text(json.dumps(summaries,indent=2))
    else:
        assert (args.output/'cc_summary.json').exists() and (args.output/'stochastic_summary.json').exists()
        freeze = args.output/'frozen_forecasts.json'
        if freeze.exists():
            raise FileExistsError('Evaluation already started; do not overwrite its freeze manifest')
        paths = sorted(args.output.glob('*.npz'))
        assert len(paths) == 36, len(paths)
        manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        freeze.write_text(json.dumps({'files':manifest,'created_unix':time.time()},indent=2))
        summaries = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(evaluation_job,(L,e,args.output)) for L,e in CASES]):
                summaries.append(future.result())
        (args.output/'evaluation_summary.json').write_text(json.dumps(summaries,indent=2))
