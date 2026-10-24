#!/usr/bin/env python3
"""New, explicitly labeled radial-eliminated analysis of frozen saved runs.

Source measurements and failed original gates are never changed. Full P
files are referenced by content hash instead of copied. Numerical q remains
uncertified; the normalized C/B expressions use no partition estimate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t as student_t
from fk4.io import write_json,source_manifest
from fk4.radial import target_overlap,coefficient_mass


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=args.source.resolve()
    summary=json.loads((source/'summary.json').read_text())
    assert summary['status']=='complete'
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    oldconfig=json.loads((source/'config.json').read_text())
    write_json(args.output/'config.json',{**oldconfig,'output':str(args.output),'source':str(source),
        'radial_elimination':True,'reinterpretation_recorded':True})
    shutil.copy2(source/'model.npz',args.output/'model.npz')
    rows=[];checks=[]
    started=time.monotonic()
    for old in summary['rows']:
        t,k=old['t'],old['k'];key=f't{t:g}_k{k}'
        frozen=np.load(source/f'{key}_frozen.npz');u,target=frozen['witness'],frozen['target']
        lam=target_overlap(u,target)
        shutil.copy2(source/f'{key}_frozen.npz',args.output/f'{key}_frozen.npz')
        if (source/f'{key}_training.json').exists():shutil.copy2(source/f'{key}_training.json',args.output/f'{key}_training.json')
        vals=[];Cs=[]
        for oldv in old['validation_rows']:
            name=f"{key}_validation_seed{oldv['seed']}"
            file=source/f'{name}.npz'
            data=np.load(file)
            factor=lam/oldv['q']
            C=factor*data['C'];C_RB=factor*data['C_RB']
            P=data['P_all_kets']
            B=coefficient_mass(u,target,P)
            # Direct P expectation and saved ladder ratio must agree. The
            # final half step is a single weighting update in these runs.
            ratio_error=abs(B-factor*oldv['B'])
            if ratio_error>1e-8:raise ArithmeticError('P mean does not match saved partition ratio; inspect intermediate temperatures')
            values={**oldv,'B':B,'rho_raw':float(np.linalg.norm(C-target)),
                'rho_RB':float(np.linalg.norm(C_RB-target)),'radial_elimination':True,
                'B_fixed_numeric_u':oldv['B'],'rho_RB_fixed_numeric_u':oldv['rho_RB'],
                'P_source_file':str(file),'P_source_key':'P_all_kets',
                'P_source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                'canonical_radius_numeric_certified':False,
                'source_old_gate_pass':summary['next_order_gate_pass'],
                'reconstruction_definition':'C_k(u_eff), u_eff=(Re<u|target>/exact_q_k(u))^(1/(2k-1))*u; not C of the stored numeric representative',
                'radial_scale_point':factor**(1/(2*k-1))}
            np.savez_compressed(args.output/f'{name}.npz',C=C,C_RB=C_RB,
                C_fixed_numeric_u=data['C'],C_RB_fixed_numeric_u=data['C_RB'],
                target=target,witness=u)
            write_json(args.output/f'{name}.json',values)
            vals.append(values);Cs.append(C_RB)
            checks.append({'key':name,'P_ratio_error':ratio_error,'rho':values['rho_RB'],'source_sha256':values['P_source_sha256']})
        Bs=np.array([r['B'] for r in vals]);half=float(student_t.ppf(.975,len(Bs)-1)*Bs.std(ddof=1)/np.sqrt(len(Bs)))
        rho=float(np.linalg.norm(np.mean(Cs,axis=0)-target))
        row={**old,'B_mean':float(Bs.mean()),'B_95_percent_halfwidth':half,
             'rho_RB_pooled':rho,'rho_RB_max':max(v['rho_RB'] for v in vals),
             'reconstruction_pass':rho<1e-3,'all_seeds_reconstruction_pass':all(v['rho_RB']<1e-3 for v in vals),
             'mass_repeat_pass':half/Bs.mean()<.01,'radial_elimination':True,
             'validation_rows':vals,'original_numeric_u_gate_pass':summary['next_order_gate_pass'],
             'full_P_files_are_hashed_external_sources':True}
        rows.append(row)
    result={**summary,'rows':rows,'source_run':str(source),'radial_elimination':True,
        'source_elapsed_seconds':summary['elapsed_seconds'],'reanalysis_seconds':time.monotonic()-started,
        'elapsed_seconds':summary['elapsed_seconds']+time.monotonic()-started,
        'next_order_gate_pass':all(r['all_seeds_reconstruction_pass'] and r['mass_repeat_pass'] for r in rows),
        'canonical_witness_definition':'Each frozen numeric representative u denotes u_eff=(Re<u|target>/exact_q_k(u))^(1/(2k-1))*u. Only the radius uses an implicit exact integral.',
        'q_error_certified':False,'original_failed_runs_preserved':True}
    write_json(args.output/'summary.json',result)
    write_json(args.output/'reanalysis_checks.json',{'checks':checks,'not_a_full_source_artifact_verifier':True})
    print([(r['t'],r['rho_RB_pooled'],r['rho_RB_max'],r['B_mean'],result['next_order_gate_pass']) for r in rows])


if __name__=='__main__':main()
