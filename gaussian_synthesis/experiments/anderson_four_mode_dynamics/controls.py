"""Bounded controls: asymmetry, U, step size, compression spacing, raw branching."""
import bootstrap
from dataclasses import replace, asdict
from pathlib import Path
import argparse
import json
import signal
from experiment import Config, run, save_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('180 s budget')))
    signal.alarm(180)
    configs = []
    for config in [Config('strang'), Config('branch_omp', cap=4), Config('schmidt'),
                   Config('measure', backend='api', discretization='sample_refit'),
                   Config('measure', backend='api_polished', discretization='two_node'),
                   Config('measure', backend='newton', discretization='two_node'),
                   Config('measure', backend='newton'), Config('unoptimized')]:
        configs.append((config, {'impurity_energy': .4}))
    for u in (0., 1., 8.):
        configs.append((Config('measure', backend='newton', discretization='two_node'), {'interaction': u}))
    for dt in (.1, .025):
        configs.append((Config('measure', backend='newton', discretization='two_node'), {'dt': dt}))
    for spacing in (1, 2, 8, 10):
        configs.append((Config('measure', backend='newton', discretization='two_node', block_steps=spacing), {}))
    for config in [Config('raw'), Config('branch_omp', cap=4), Config('measure', backend='newton', discretization='two_node')]:
        configs.append((config, {'final_time': .6}))
    summaries = []
    for index, (config, kwargs) in enumerate(configs):
        output = run(config, **kwargs)
        save_case(args.output, index, output)
        summaries.append(output[0])
        (args.output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
        print(json.dumps({'index': index, 'method': config.method, 'backend': config.backend,
                          'kwargs': kwargs, 'block': config.block_steps,
                          'seconds': output[0]['total_seconds'], 'infidelity': output[0]['max_infidelity'],
                          'cost': output[0]['final_coefficient_cost']}), flush=True)
    signal.alarm(0)


if __name__ == '__main__':
    main()
