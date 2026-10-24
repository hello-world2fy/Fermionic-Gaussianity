#!/usr/bin/env python3
"""Attach conservative exact-Gaussian repair costs to independent columns."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4.io import write_json,source_manifest
from fk4.rounding import corrected_cost
from fk4.state_enclosure import enclose_ground_state
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--columns',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert json.loads((args.columns/'verification.json').read_text())['artifact_verification_passed']
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    best=json.loads((args.columns/'best_costs.json').read_text())['rows'];model=build_model()
    started=time.monotonic();rows=[]
    for r in best:
        path=args.columns/(r['key']+'.npz');a=np.load(path);t=r['t'];target=model.target(t)
        enclosure=enclose_ground_state(model,t,target)
        bound=corrected_cost(a['finite_kets'],a['finite_coefficients'],target,model.j_matrices,
                             enclosure['saved_target_to_exact_ground_state_distance_upper'])
        rows.append({'t':t,'original_key':r['key'],'original_floating_cost':r['U'],
            'source_file':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'physical_target_enclosure':enclosure,**bound})
        print(t,r['U'],bound['corrected_Gaussian_cost_upper'],flush=True)
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,
        'elapsed_seconds':time.monotonic()-started,'global_optimality_claimed':False,
        'proof_file':'GAUSSIAN_ROUNDING_CERTIFICATES.md',
        'construction':'Each numerical branch is replaced by a nearby exact Gaussian as defined by its covariance; exact Fock residual included. Original saved arrays and costs remain unchanged.'})
    lines=['# 独立构造的保守精确Gaussian修补成本','',
        '原有有限分支为浮点Gaussian近似。本证书通过精确协方差定义相邻的真正Gaussian态，并将分支移动及精确物理目标误差计入完整Fock修补成本。不是把原浮点分支直接称为精确Gaussian；证明与明确的IEEE舍入假设见GAUSSIAN_ROUNDING_CERTIFICATES.md。','',
        '| t | 原浮点U | 修补构造的严格成本上界 | Gaussian移动修补 | 实际主分支数 + Fock修补上限 |',
        '|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['t']:g} | {r['original_floating_cost']:.10g} | {r['corrected_Gaussian_cost_upper']:.10g} | {r['Gaussian_atom_repair_cost_upper']:.3g} | {r['main_actual_branch_count']} + 128 |")
    lines+=['','t=0另有精确成本2的解析构造，应采用更强的解析值。其他点的修补成本是完整Gaussian字典的可行上界，不是最优性声明。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
