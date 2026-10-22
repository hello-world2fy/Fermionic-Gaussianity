"""Small, serial matched-accuracy benchmark with complete costs and failures."""
import bootstrap
import argparse
from dataclasses import replace, asdict
from pathlib import Path
import json
import time
import signal
import numpy as np
from gaussian_synthesis.solver import peak_memory_mib
from experiment import Config, run, save_case, configs

FIDELITY_THRESHOLD = 1e-5
DOUBLE_THRESHOLD = 2e-4
SEEDS = (20261023, 20261024, 20261025)


def label(config):
    if config.method == 'measure':
        return f'{config.backend}:{config.discretization}'
    return config.method+(':'+config.discretization if config.method in ('haar', 'unoptimized') else '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('180 second experiment budget')))
    signal.alarm(180)
    # Warm-up only tiny linear algebra, never choose parameters by reference error.
    run(Config('strang'), final_time=.1)
    summaries = []
    grouped = []
    index = 0
    for config in configs():
        cases = []
        # Same seeds across stochastic methods; three serial runs, no best-seed selection.
        for seed in SEEDS:
            output = run(replace(config, seed=seed))
            save_case(args.output, index, output)
            summary = output[0]
            summary['case_index'] = index
            cases.append(summary)
            summaries.append(summary)
            index += 1
        passed = all(c['max_infidelity'] <= FIDELITY_THRESHOLD and
                     c['max_double_error'] <= DOUBLE_THRESHOLD for c in cases)
        record = {'label': label(config), 'config': asdict(config), 'case_indices': [c['case_index'] for c in cases],
                  'matched_error_pass_all_seeds': passed,
                  'median_total_seconds': float(np.median([c['total_seconds'] for c in cases])),
                  'time_range': [min(c['total_seconds'] for c in cases), max(c['total_seconds'] for c in cases)],
                  'max_infidelity_all_seeds': max(c['max_infidelity'] for c in cases),
                  'max_double_error_all_seeds': max(c['max_double_error'] for c in cases),
                  'final_atoms': [c['final_atoms'] for c in cases],
                  'peak_branch_atoms': max(c['peak_branch_atoms'] for c in cases),
                  'final_coefficient_cost': [c['final_coefficient_cost'] for c in cases],
                  'max_coefficient_cost': [c.get('max_coefficient_cost') for c in cases],
                  'mean_postcompression_cost': [c.get('mean_postcompression_cost') for c in cases],
                  'max_postcompression_cost': [c.get('max_postcompression_cost') for c in cases],
                  'solver_calls_each': [c['original_solver_calls'] for c in cases],
                  'sample_counts_each': [{key: c[key] for key in ('training_samples', 'diagnostic_samples',
                                         'validation_samples', 'sample_proposals', 'mass_samples')} for c in cases],
                  'median_timing_components': {key: float(np.median([c['timing_components'][key] for c in cases]))
                                                for key in cases[0]['timing_components']}}
        grouped.append(record)
        (args.output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
        (args.output/'comparison.json').write_text(json.dumps(grouped, indent=2, allow_nan=False))
        print(json.dumps({key: record[key] for key in ('label', 'matched_error_pass_all_seeds',
              'median_total_seconds', 'max_infidelity_all_seeds', 'final_coefficient_cost')}), flush=True)
    metadata = {'seeds': SEEDS, 'mode_count': 4, 'interaction': 4, 'final_time': 2, 'dt': .05,
                'compression_every_steps': 4, 'repetitions': 3, 'parallelism': 'none; library threads=1',
                'accuracy_thresholds': {'max_infidelity': FIDELITY_THRESHOLD, 'max_double_error': DOUBLE_THRESHOLD},
                'threshold_scope': 'Fixed before this comparison after a pilot; exploratory, no statistical certification',
                'peak_rss_mib': peak_memory_mib()}
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    signal.alarm(0)


if __name__ == '__main__':
    main()
