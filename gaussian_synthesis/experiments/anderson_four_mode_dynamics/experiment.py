"""Closed-loop propagation. Exact trajectories are only computed afterwards."""
import bootstrap
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import json
import csv
import numpy as np
from model import Anderson, GaussianSum, PHYSICAL, gaussian_defect, schmidt_decompose, unitary
from bridge import solve_measure, sampled_discretization, two_node_discretization, symmetry_dictionary, refit
from gaussian_synthesis.measure import GaussianCoefficientMeasure
from model import BASIS


@dataclass(frozen=True)
class Config:
    method: str
    backend: str = 'newton'
    discretization: str = 'sample_refit'
    samples: int = 128
    cap: int = 8
    block_steps: int = 4
    seed: int = 20261023


def run(config, *, interaction=4., dt=.05, final_time=2., impurity_energy=0.):
    total_start = time.perf_counter()
    start = time.perf_counter()
    model = Anderson(interaction=interaction, impurity_energy=impurity_energy, dt=dt)
    steps = round(final_time/dt)
    if steps*dt != final_time and abs(steps*dt-final_time)>1e-12:
        raise ValueError('final_time must be a multiple of dt')
    if config.method == 'raw' and steps > 12 and interaction:
        raise ValueError('Raw branches are explicitly limited to 4096')
    state = model.initial_sum()
    vector = model.initial.copy()
    exact_step = unitary(model.physical_h, dt) if config.method == 'exact' else None
    previous_u = None
    free_block = unitary(model.free, config.block_steps*dt)
    setup_seconds = time.perf_counter()-start
    rows, snapshots, events = [], [], []
    times = {'propagation': 0., 'solve': 0., 'sampling': 0., 'moment_compression': 0.,
             'symmetry': 0., 'refit': 0., 'observables_and_recording': 0.}
    cumulative_residual = 0.
    peak_atoms = 1
    for step in range(steps+1):
        if step:
            start = time.perf_counter()
            if config.method == 'exact':
                vector[PHYSICAL.ravel()] = exact_step @ vector[PHYSICAL.ravel()]
            elif config.method == 'strang':
                vector = model.strang @ vector
            else:
                state = model.branch(state)
                peak_atoms = max(peak_atoms, state.count)
            times['propagation'] += time.perf_counter()-start
            if (config.method not in ('exact', 'strang', 'raw') and interaction != 0
                    and (step % config.block_steps == 0 or step == steps)):
                event = {'step': step, 'input_atoms': state.count}
                start = time.perf_counter()
                target = state.vector()
                times['refit'] += time.perf_counter()-start
                if config.method == 'schmidt':
                    start = time.perf_counter()
                    state = schmidt_decompose(target)
                    times['refit'] += time.perf_counter()-start
                elif config.method in ('branch_omp', 'weight'):
                    start = time.perf_counter()
                    if config.method == 'weight':
                        selected = np.argsort(-np.abs(state.coefficients))[:config.cap]
                        state = GaussianSum(state.atoms[:, selected], state.coefficients[selected])
                    else:
                        state, info = refit(state, target, config.cap)
                        event.update(info)
                    times['refit'] += time.perf_counter()-start
                else:
                    seed = config.seed+1009*step
                    if config.method == 'measure':
                        initial = free_block @ previous_u if previous_u is not None else None
                        measure, info = solve_measure(target, config.backend, seed, initial)
                        previous_u = measure.u.copy()
                        times['solve'] += info['solve_seconds']
                        event.update(info)
                        event['witness'] = [[z.real, z.imag] for z in measure.u]
                        if config.discretization == 'two_node':
                            state, info = two_node_discretization(measure)
                        else:
                            state, info = sampled_discretization(measure, config.samples, seed+1)
                    elif config.method == 'unoptimized':
                        measure = GaussianCoefficientMeasure(BASIS, 2, target)
                        state, info = sampled_discretization(measure, config.samples, seed+1)
                        event['ablation'] = 'u=target, no convex solve'
                    elif config.method == 'direct_quadric':
                        state, info = two_node_discretization(GaussianCoefficientMeasure(BASIS, 2, target))
                        start = time.perf_counter()
                        state, correction = refit(state, target, 2)
                        times['refit'] += time.perf_counter()-start
                        info.update(correction)
                        event['ablation'] = 'four-mode geometric two-atom decomposition without convex solve'
                    elif config.method == 'haar':
                        state, info = sampled_discretization(None, config.samples, seed+1, importance=False)
                    else:
                        raise ValueError(config.method)
                    times['sampling'] += info['sampling_seconds']
                    times['moment_compression'] += info.get('moment_compression_seconds', 0.)
                    event.update(info)
                    event['raw_discretization_error'] = float(np.linalg.norm(state.vector()-target))
                    if config.discretization == 'symmetry_refit':
                        start = time.perf_counter()
                        state = symmetry_dictionary(state)
                        times['symmetry'] += time.perf_counter()-start
                    if config.discretization in ('sample_refit', 'symmetry_refit') and config.method != 'direct_quadric':
                        start = time.perf_counter()
                        state, info = refit(state, target, config.cap)
                        times['refit'] += time.perf_counter()-start
                        event.update(info)
                residual = float(np.linalg.norm(target-state.vector()))
                cumulative_residual += residual
                event.update(local_residual=residual, output_atoms=state.count,
                             coefficient_cost=state.cost())
                events.append(event)
        start = time.perf_counter()
        if config.method not in ('exact', 'strang'):
            vector = state.vector()
        norm2 = float(np.vdot(vector, vector).real)
        probabilities = np.abs(vector)**2/norm2
        row = {'step': step, 'time': step*dt, 'norm': float(np.sqrt(norm2)),
               'double': float(model.double @ probabilities), 'density': float(model.density @ probabilities),
               'cumulative_residual': cumulative_residual,
               'atoms': state.count if config.method not in ('exact', 'strang') else 0,
               'coefficient_cost': state.cost() if config.method not in ('exact', 'strang') else None}
        snapshots.append(vector.copy())
        rows.append(row)
        times['observables_and_recording'] += time.perf_counter()-start
    total_seconds = time.perf_counter()-total_start
    times['other_loop_overhead'] = total_seconds-setup_seconds-sum(times.values())
    # Reference trajectory is created only AFTER timing and propagation finish.
    energies, eig = np.linalg.eigh(model.hamiltonian)
    times_grid = np.arange(steps+1)*dt
    exact = (eig @ (np.exp(-1j*energies[:, None]*times_grid)*
                   (eig.conj().T @ model.initial)[:, None])).T
    split = model.initial.copy()
    for step, (row, vector) in enumerate(zip(rows, snapshots)):
        if step:
            split = model.strang @ split
        normalized = vector/np.linalg.norm(vector)
        overlap = np.vdot(exact[step], normalized)
        ep = np.abs(exact[step])**2
        row.update(infidelity=float(max(0., 1-abs(overlap)**2)),
                   compression_distance=float(np.linalg.norm(vector-split)),
                   double_error=float(abs(row['double']-model.double @ ep)),
                   density_error=float(abs(row['density']-model.density @ ep)),
                   trotter_distance=float(np.linalg.norm(exact[step]-split)),
                   exact_double=float(model.double @ ep))
    summary = {'config': asdict(config), 'interaction': interaction, 'dt': dt,
               'final_time': final_time, 'impurity_energy': impurity_energy,
               'total_seconds': total_seconds, 'setup_seconds': setup_seconds,
               'timing_components': times, 'peak_branch_atoms': peak_atoms,
               'final_atoms': rows[-1]['atoms'], 'final_coefficient_cost': rows[-1]['coefficient_cost'],
               'original_solver_calls': sum('source_function' in event for event in events),
               'sample_proposals': sum(event.get('proposal_samples', 0) for event in events),
               'mass_samples': sum(event.get('mass_samples', 0) for event in events),
               'training_samples': sum(event.get('training_samples', 0) for event in events),
               'diagnostic_samples': sum(event.get('diagnostic_samples', 0) for event in events),
               'validation_samples': sum(event.get('validation_samples', 0) for event in events),
               'final_gaussian_defect': gaussian_defect(state.atoms) if config.method not in ('exact', 'strang') else None,
               'final_cumulative_residual': cumulative_residual,
               'max_atoms_after_compression': max((event['output_atoms'] for event in events), default=state.count),
               'final': rows[-1]}
    for name in ('infidelity', 'double_error', 'density_error', 'compression_distance'):
        summary['max_'+name] = max(row[name] for row in rows)
    if config.method not in ('exact', 'strang'):
        summary['max_coefficient_cost'] = max(row['coefficient_cost'] for row in rows)
        summary['mean_postcompression_cost'] = float(np.mean([e['coefficient_cost'] for e in events])) if events else state.cost()
        summary['max_postcompression_cost'] = max((e['coefficient_cost'] for e in events), default=state.cost())
    return summary, rows, events, state, np.array(snapshots)


def save_case(directory, index, output):
    summary, rows, events, state, snapshots = output
    prefix = directory/f'case_{index:03d}'
    with prefix.with_suffix('.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix('.json').write_text(json.dumps({'summary': summary, 'events': events}, indent=2,
                                                    allow_nan=False))
    arrays = {'snapshots': snapshots}
    if summary['config']['method'] not in ('exact', 'strang'):
        arrays.update(atoms=state.atoms, coefficients=state.coefficients)
    np.savez_compressed(str(prefix)+'_states.npz', **arrays)


def configs():
    return [Config('exact'), Config('strang'), Config('schmidt'),
            Config('branch_omp', cap=4), Config('weight', cap=4),
            Config('haar'), Config('measure', backend='api'),
            Config('measure', backend='sgd'), Config('measure', backend='newton'),
            Config('measure', backend='newton', discretization='sample_raw'),
            Config('measure', backend='sgd', discretization='two_node'),
            Config('measure', backend='newton', discretization='two_node'),
            Config('measure', backend='newton', discretization='symmetry_refit', samples=64, cap=4),
            Config('haar', discretization='symmetry_refit', samples=64, cap=4),
            Config('unoptimized'), Config('direct_quadric', discretization='two_node'),
            Config('measure', backend='sgd_polished', discretization='two_node'),
            Config('measure', backend='api_polished', discretization='two_node')]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for index, config in enumerate(configs()):
        output = run(config)
        save_case(args.output, index, output)
        summaries.append(output[0])
        (args.output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
        print(json.dumps({'index': index, 'config': asdict(config), 'seconds': output[0]['total_seconds'],
                          'infidelity': output[0]['max_infidelity'], 'cost': output[0]['final_coefficient_cost'],
                          'calls': output[0]['original_solver_calls']}), flush=True)
