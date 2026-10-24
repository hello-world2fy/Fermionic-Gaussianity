#!/usr/bin/env python3
"""Directed, conservative N_k upper bounds via exact Cartan block inclusion."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk4.io import source_manifest,write_json
from fk4.exact_cartan import ExactCartan
from fk4.certificates import _bounds
from fk4.state_enclosure import enclose_ground_state
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--directions',type=Path,nargs='+',required=True)
    p.add_argument('--moment',type=Path,default=Path('results/cartan_exact_k4_01'))
    args=p.parse_args()
    assert json.loads((args.moment/'verification.json').read_text())['artifact_verification_passed']
    exact=ExactCartan(args.moment/'proof.json')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    model=build_model();enclosures={};rows=[]
    for path in args.directions:
        match=re.fullmatch(r't([0-9.]+)_k([0-9]+)_frozen.npz',path.name)
        if not match:raise ValueError('unexpected frozen-file name')
        t,k=float(match[1]),int(match[2])
        if k%exact.k:raise ValueError('target order must be an integer multiple of anchor order')
        data=np.load(path);z,target=data['z'],data['target']
        if t not in enclosures:enclosures[t]=enclose_ground_state(model,t,target)
        c=exact.certificate(z,target);safe=c.pop('z_feasible')
        iv.dps=70
        high_upper=_bounds(iv.mpf(c['q_normalized_interval'][1])**(k//exact.k))[1]
        assert high_upper<=1
        L=_bounds(iv.mpf(c['L_certified_saved_target'])-iv.mpf(c['witness_hilbert_norm_upper'])*iv.mpf(enclosures[t]['saved_target_to_exact_ground_state_distance_upper']))[0]
        key=f't{t:g}_k{k}'
        np.savez_compressed(args.output/f'{key}_certificate.npz',direction_z=z,z_feasible=safe,target=target)
        rows.append({'t':t,'direction_order':k,'target_moment_order':k,'anchor_order':exact.k,
            'source_file':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'anchor_moment_interval':c['q_normalized_interval'],'high_order_normalized_q_upper':high_upper,
            'L_certified_exact_physical_target':L,'full_Gaussian_dictionary_feasible':True,
            'Nk_upper_proof':'Pi_k <= tensor_(k/4) Pi_4, hence q_k <= q4^(k/4) and Nk <= N4',
            'point_estimate_of_canonical_L_certified':False,'bound_is_conservative':True,
            'normalization_moment_proof_file':str((args.moment/'proof.json').resolve())})
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'physical_target_enclosures':enclosures,
        'full_Gaussian_dictionary':True,'higher_Nk_upper_bounds_certified':True,
        'high_order_integrals_directly_evaluated':False,'tight_duality_gaps_claimed':False})
    lines=['# 高阶witness的保守Cartan分块证书','',
      '利用完整Gaussian coherent projector的嵌入 Pi_(a+b)≤Pi_a⊗Pi_b，严格有 q_(a+b)≤q_a q_b。因此每个4的整数倍阶数满足 N_k≤N_4。这里用已独立核验的精确k=4整数矩及有向舍入归一化，并转移到包围的精确物理基态。','',
      '这些是真实Nk上界与对偶可行证书，但通常很松。表中L不能替代高阶Monte Carlo canonical L点估计，也不能据此宣称小的最优性缺口。','',
      '| t | 方向/目标阶数k | 认证来源阶数 | 归一化后q_k严格上界 | L严格下界（精确物理态） |',
      '|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['t']:g} | {r['direction_order']} | {r['anchor_order']} | {r['high_order_normalized_q_upper']:.17g} | {r['L_certified_exact_physical_target']:.10g} |")
    lines+=['','证明见CARTAN_BLOCK_BOUNDS.md。未假设方向已经达到canonical最优解；可行性对任何给定方向成立。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
