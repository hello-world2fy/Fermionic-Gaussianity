#!/usr/bin/env python3
"""Finalize the declared E1 scan and certificates after every source passes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4.io import write_json,source_manifest
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);args=p.parse_args()
    config=json.loads(args.config.read_text())
    if args.state.exists():raise FileExistsError('inspect existing finalization state')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='12')
    completed=[]
    def state(status,**extra):write_json(args.state,{'status':status,'config':str(args.config),'completed_commands':completed,**extra})
    def run(command):
        state('running',command=command);subprocess.run([sys.executable,*command],check=True,env=env)
        completed.append(command);state('command_complete')
    try:
        while True:
            pending=[]
            for directory in config['runs']:
                path=Path(directory);sf=path/'summary.json';vf=path/'verification.json'
                if not sf.exists():pending.append(directory);continue
                s=json.loads(sf.read_text())
                if s['status']!='complete':pending.append(directory);continue
                if not s['next_order_gate_pass']:
                    state('stopped_at_failed_gate',failed_source=directory);raise SystemExit(2)
                if not vf.exists() and directory in config['verify_when_ready']:
                    run(['scripts/verify_results.py',directory]);run(['scripts/make_report.py',directory])
                if not vf.exists():pending.append(directory);continue
                assert json.loads(vf.read_text())['artifact_verification_passed']
            if not pending:break
            state('waiting_for_verified_sources',pending=pending);time.sleep(5)
        suffix=config['suffix'];endpoint=Path(config['endpoint'])
        assert json.loads((endpoint/'verification.json').read_text())['artifact_verification_passed']
        # The endpoint direction is exact at every order, so materialize only
        # its small coordinate files for the generic block-certificate reader.
        directory=Path(f'results/endpoint_block_directions_{suffix}');directory.mkdir(exist_ok=False)
        write_json(directory/'manifest.json',source_manifest(directory/'source_snapshot'))
        source=endpoint/'finite_projector.npz';target=np.load(source)['target'];model=build_model();z=model.S.conj().T@target
        write_json(directory/'provenance.json',{'source_file':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'meaning':'The exact fixed t0 canonical direction, reused at orders16/32/64/128; not a new optimization or sampling run.'})
        directions=[]
        for path in map(Path,config['runs']):
            for f in path.glob('*_frozen.npz'):
                k=int(f.stem.split('_k')[1].split('_')[0])
                if k>=8:directions.append(str(f))
        for k in [16,32,64,128]:
            f=directory/f't0_k{k}_frozen.npz';np.savez_compressed(f,z=z,witness=model.S@z,target=target);directions.append(str(f))
        blocks=f'results/block_Nk_certificates_final_{suffix}'
        run(['scripts/certify_block_witnesses.py','--output',blocks,'--directions',*directions])
        run(['scripts/verify_witness_certificates.py',blocks])
        windows=f'results/energy_window_witness_certificates_final_{suffix}'
        lower=[str(f) for k in [2,4] for f in Path(f'results/cartan_k{k}_certified_01').glob('*_frozen.npz')]
        run(['scripts/certify_energy_window_witnesses.py','--output',windows,'--cost-certificates',config['cost_certificates'],
             '--directions',*lower,*directions])
        run(['scripts/verify_witness_certificates.py',windows]);run(['scripts/plot_window_certificates.py',windows])
        figure=f'results/hierarchy_final_{suffix}'
        run(['scripts/figure_hierarchy.py','--final','--output',figure,'--columns',config['columns'],
             '--cost-certificates',config['cost_certificates'],'--endpoint',config['endpoint'],'--block-certificates',blocks,'--runs',*config['runs']])
        composition=f'results/canonical_composition_final_{suffix}'
        run(['scripts/report_canonical_composition.py','--output',composition,'--endpoint',config['endpoint'],'--runs',*config['runs']])
        state('complete',blocks=blocks,windows=windows,figure=figure,composition=composition)
    except subprocess.CalledProcessError as exc:
        state('child_failed',command=exc.cmd,exit_code=exc.returncode);raise


if __name__=='__main__':main()
