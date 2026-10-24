#!/usr/bin/env python3
"""Wait for the declared E1 sources, then freeze and verify the E4/E5 matrix."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk4.io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);args=p.parse_args()
    config=json.loads(args.config.read_text())
    if args.state.exists():raise FileExistsError('inspect the previous pipeline before restarting')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='12')
    completed=[]
    def state(status,**extra):write_json(args.state,{'status':status,'config':str(args.config),'completed_commands':completed,**extra})
    def run(command):
        state('running',command=command)
        subprocess.run([sys.executable,*command],check=True,env=env)
        completed.append(command);state('command_complete')
    while True:
        pending=[]
        for directories in config['sources'].values():
            for directory in directories:
                path=Path(directory);sfile=path/'summary.json';vfile=path/'verification.json'
                if not sfile.exists():pending.append(directory);continue
                summary=json.loads(sfile.read_text())
                if summary['status']!='complete':pending.append(directory);continue
                if not summary['next_order_gate_pass']:
                    state('stopped_at_failed_source_gate',failed_source=directory);raise SystemExit(2)
                if not vfile.exists():pending.append(directory);continue
                if not json.loads(vfile.read_text())['artifact_verification_passed']:
                    state('stopped_at_failed_artifact_verification',failed_source=directory);raise SystemExit(3)
        if not pending:break
        state('waiting_for_verified_sources',pending=pending)
        time.sleep(5)
    try:
        indices=[]
        for order,directories in config['sources'].items():
            output=f"results/fk_k{order}_all_transfer_index_{config['suffix']}"
            run(['scripts/assemble_verified_runs.py','--runs',*directories,'--output',output]);indices.append(output)
        frozen=f"results/frozen_dictionaries_k2_k8_k16_k32_{config['suffix']}"
        predictions=f"results/frozen_predictions_k2_k8_k16_k32_{config['suffix']}"
        run(['scripts/freeze_dictionaries.py','--output',frozen,'--columns',config['columns'],
             '--replica-runs',config['k2_source'],*indices])
        run(['scripts/run_frozen_predictions.py','--frozen',frozen,'--output',predictions])
        run(['scripts/verify_frozen_predictions.py',predictions])
        run(['scripts/report_frozen_predictions.py',predictions])
        run(['scripts/audit_prediction_costs.py',predictions,'--prior-benchmark',config['prior_benchmark']])
        state('complete',frozen=frozen,predictions=predictions)
    except subprocess.CalledProcessError as exc:
        state('child_failed',command=exc.cmd,exit_code=exc.returncode);raise


if __name__=='__main__':main()
