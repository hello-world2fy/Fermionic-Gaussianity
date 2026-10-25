#!/usr/bin/env python3
"""Screen the late count-coherence return against frozen collective models.

The original G=4 trajectory has already been inspected. This is a pilot,
not an unseen-parameter confirmation. All model trajectories are nonetheless
saved before exact propagation of their own initial states.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.frozen import channel_basis, propagate
from pairing_formation.linear import squeezed_pair_matrix
from pairing_formation.sectors import transfer_operator
from pairing_formation.coherent_return import factor_observables, dephase_groups, flat_band_hamiltonian


def job(task):
    L, source, q, output = task
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(L, 4.)
    training = model.evolve([1.])[0]
    if source == 'rpa':
        T = squeezed_pair_matrix(model, [1.])[0]
    elif source == 'direct':
        T = training[model.blocks[1][1]]
    elif source == 'witness':
        T = np.load(root/'replica2_L12_G4_return_01/t1.npz')['witness'][model.blocks[1][1]]
    B, singular = channel_basis(model, T, q)
    c = B.conj().T@training
    initial_fidelity = float(np.linalg.norm(c)**2)
    c /= np.sqrt(initial_fidelity)
    times = np.linspace(0., 2.5, 51)
    H = B.conj().T@(model.H@B)
    states = propagate(H, c, times)@B.T
    name = f'L{L}_{source}_q{q}'
    frozen = output/(name+'_frozen.npz')
    np.savez_compressed(frozen, basis=B, coefficients=c, training=training, energies=model.energies)
    prediction = output/(name+'_prediction.npz')
    np.savez_compressed(prediction, times=times, states=states)
    record = {'name': name, 'L': L, 'source': source, 'q': q, 'dimension': B.shape[1],
              'initial_fidelity': initial_fidelity, 'singular_values': singular.tolist(),
              'orthogonality_error': float(np.linalg.norm(B.conj().T@B-np.eye(B.shape[1]))),
              'frozen_file': frozen.name, 'frozen_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
              'prediction_file': prediction.name,
              'prediction_sha256': hashlib.sha256(prediction.read_bytes()).hexdigest()}
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {'scope': 'Retrospective mechanism pilot on the inspected G=4 trajectory, not new holdout conditions.',
                'L': [12, 16], 'G': 4., 'training_time': 1., 'elapsed': [0., 2.5],
                'channel_counts': [1, 2, 3], 'primary_elapsed': 2., 'theta': .2,
                'sources': ['early direct m=1 amplitude', 'early RPA matrix', 'early canonical witness m=1 amplitude (L12 only)'],
                'models': 'Frozen onsite Gaussian-generated channel spans, projected-Hamiltonian propagation; these are not actual canonical samples.',
                'controls': 'Exact evolution from every model initial state; vacuum/rest and complete transfer-count dephasing of the exact training state; exactly collective flat-band chains.',
                'interpretation_rule': 'Do not infer a canonical-specific mechanism if conventional RPA channels predict the late signal as accurately, or if an exactly one-direction flat-band model exhibits it.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    tasks = [(L, source, q, args.output) for L in (12, 16)
             for source in (('direct', 'rpa', 'witness') if L == 12 else ('direct', 'rpa')) for q in (1, 2, 3)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job, task) for task in tasks]):
            records.append(future.result())
    # Boundary between frozen predictions and reference access.
    (args.output/'manifest.json').write_text(json.dumps({'protocol': protocol, 'records': records}, indent=2))
    evaluations, controls = [], []
    for L in (12, 16):
        model = picket_fence(L, 4.)
        transfer = transfer_operator(model)
        training = model.evolve([1.])[0]
        group = [r for r in records if r['L'] == L]
        times = np.load(args.output/group[0]['prediction_file'])['times']
        initial = np.column_stack([training]+[np.load(args.output/r['prediction_file'])['states'][0] for r in group])
        exact = model.evolve(times, initial)
        np.savez_compressed(args.output/f'L{L}_references.npz', times=times, states=exact)
        for j, record in enumerate(group):
            states = np.load(args.output/record['prediction_file'])['states']
            rows = []
            for t, pred, ref in zip(times, states, exact):
                truth, own = ref[:, 0], ref[:, j+1]
                row = factor_observables(model.m, pred)
                true = factor_observables(model.m, truth)
                own_obs = factor_observables(model.m, own)
                rows.append({'elapsed': float(t), 'predicted': row, 'exact': true,
                             'fidelity': float(abs(np.vdot(truth, pred))**2),
                             'count_TV': float(np.sum(abs(np.array(row['count'])-true['count']))/2),
                             'echo_error': row['echo_excess']-true['echo_excess'],
                             'initial_only_echo_error': own_obs['echo_excess']-true['echo_excess'],
                             'dynamics_only_echo_error': row['echo_excess']-own_obs['echo_excess'],
                             'cross_error': float(abs(np.vdot(pred, transfer@pred)-np.vdot(truth, transfer@truth)))})
            evaluations.append({'model': record, 'rows': rows})
            last = next(r for r in rows if r['elapsed'] == 2.)
            print(json.dumps({'name':record['name'],'elapsed':2.,'echo_error':last['echo_error'],
                              'count_TV':last['count_TV'],'fidelity':last['fidelity']}), flush=True)
        for name, factors in [('pure', training[:, None]),
                              ('vacuum_rest_dephased', dephase_groups(training, model.m > 0)),
                              ('count_dephased', dephase_groups(training, model.m))]:
            evolved = model.evolve(times, factors)
            rows = [{'elapsed': float(t), **factor_observables(model.m, V)} for t, V in zip(times, evolved)]
            controls.append({'L': L, 'control': name, 'rows': rows,
                             'initial_collective_cross': [float(x) for x in
                                 (np.vdot(factors, transfer@factors).real, np.vdot(factors, transfer@factors).imag)]})
    flat = []
    for L in (8, 12, 16, 32, 64, 128):
        r=L//2
        H=flat_band_hamiltonian(r,4.)
        init=np.zeros(r+1,complex);init[0]=1.
        times=np.linspace(0.,6.,601)
        states=expm_multiply(-1j*H,init,start=0,stop=6.,num=len(times),traceA=-1j*H.diagonal().sum())
        rows=[{'time':float(t),**factor_observables(np.arange(r+1),s)} for t,s in zip(times,states)]
        flat.append({'L':L,'G':4.,'rows':rows,'scope':'Exactly one collective orbital direction per band; exploratory size comparison.'})
    (args.output/'evaluation.json').write_text(json.dumps(evaluations, indent=2))
    (args.output/'phase_controls.json').write_text(json.dumps(controls, indent=2))
    (args.output/'flat_band_controls.json').write_text(json.dumps(flat, indent=2))
