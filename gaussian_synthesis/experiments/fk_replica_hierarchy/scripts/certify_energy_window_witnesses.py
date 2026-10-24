#!/usr/bin/env python3
"""Compare candidate replica directions using rigorous q4 normalization.

High-k direction does not imply high-k moment certification. The exact q4
certificate and physical-target enclosure establish all displayed inequalities.
U is a declared conditional cost threshold, never an inferred optimal cost.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk4.io import write_json,source_manifest
from fk4.exact_cartan import ExactCartan
from fk4.state_enclosure import enclose_ground_state
from fk4.energy_windows import hopping_eigenbasis
from fk4.window_certificates import restricted_support,necessity
from fk4.certificates import _bounds
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--directions',type=Path,nargs='+',required=True)
    p.add_argument('--moment',type=Path,default=Path('results/cartan_exact_k4_01'))
    p.add_argument('--cost-thresholds',type=float,nargs='+',default=[2.])
    p.add_argument('--cost-certificates',type=Path,help='Also evaluate the independently verified constructive upper for each t')
    p.add_argument('--t',type=float,nargs='+',default=[9,9.1,11,20])
    args=p.parse_args()
    costs={}
    if args.cost_certificates:
        assert json.loads((args.cost_certificates/'verification.json').read_text())['artifact_verification_passed']
        costs={r['t']:r for r in json.loads((args.cost_certificates/'summary.json').read_text())['rows']}
    assert json.loads((args.moment/'verification.json').read_text())['artifact_verification_passed']
    exact=ExactCartan(args.moment/'proof.json');assert exact.k==4
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{k:[str(x) for x in v] if k=='directions' else str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    model=build_model();rows=[];witnesses=[];enclosures={}
    for t in args.t:
        target=model.target(t)
        enclosure=enclose_ground_state(model,t,target)
        enclosures[t]=enclosure
        candidates=[]
        for path in args.directions:
            match=re.fullmatch(r't([0-9.]+)_k([0-9]+)_frozen.npz',path.name)
            if match and float(match[1])==t:
                data=np.load(path)
                if np.linalg.norm(data['target']-target)>1e-12:raise ValueError('target mismatch')
                candidates.append((int(match[2]),data['z'],str(path.resolve()),hashlib.sha256(path.read_bytes()).hexdigest()))
        filled=hopping_eigenbasis()[:,2].astype(complex)
        overlap=np.vdot(model.S@filled,target)
        filled*=overlap/abs(overlap)
        candidates.append((0,filled,'filled-hopping-reference',None))
        for order,z,source,digest in candidates:
            certificate=exact.certificate(z,target)
            safe=certificate.pop('z_feasible')
            iv.dps=70
            L=iv.mpf(certificate['L_certified_saved_target'])-iv.mpf(certificate['witness_hilbert_norm_upper'])*iv.mpf(enclosure['saved_target_to_exact_ground_state_distance_upper'])
            Llower=_bounds(L)[0]
            name=f't{t:g}_source_k{order}'
            np.savez_compressed(args.output/f'{name}.npz',direction_z=z,z_feasible=safe,target=target)
            witnesses.append({'key':name,'t':t,'source_direction_order':order,'normalization_order':4,
                'source_file':source,'source_sha256':digest,'L_exact_physical_ground_lower':Llower,
                'q_normalized_interval':certificate['q_normalized_interval'],
                'full_Gaussian_support_upper':1.,'candidate_canonical_fit_not_needed_for_validity':True,
                'high_order_moment_not_certified':order>4})
            for delta in np.logspace(-5,0,21):
                bound=restricted_support(safe,t,float(delta))
                thresholds=[(U,'declared_conditional_threshold') for U in args.cost_thresholds]
                if t in costs:thresholds.append((costs[t]['corrected_Gaussian_cost_upper'],'certified_constructive_upper'))
                for U,kind in thresholds:
                    rows.append({'witness_key':name,'t':t,'source_direction_order':order,'normalization_order':4,
                        'delta':float(delta),'declared_cost_threshold':U,'L_physical_lower':Llower,
                        'cost_threshold_kind':kind,
                        'cost_certificate_source':str(args.cost_certificates.resolve()) if kind=='certified_constructive_upper' else None,
                        **bound,**necessity(Llower,bound['restricted_support_upper'],U),
                        'meaning':'For every exact Gaussian decomposition of the exact phase-aligned ground state with cost B<=U, coefficient mass fraction inside E<=E_G+delta obeys this upper bound.',
                        'existence_of_cost_U_decomposition_not_asserted_by_this_certificate':kind!='certified_constructive_upper'})
    write_json(args.output/'summary.json',{'status':'complete','witnesses':witnesses,'rows':rows,
        'physical_target_enclosures':enclosures,'full_Gaussian_dictionary':True,
        'normalization_uses_exact_order4_only':True,'high_order_N_certificate_claimed':False,
        'all_U_values_are_declared_conditional_thresholds':not bool(costs),
        'constructive_cost_certificates_included':bool(costs)})
    lines=['# Replica方向与能窗组成必要性','',
        '对不同阶数得到的方向，统一使用精确k=4矩的区间上界归一化；因此这些是完整Gaussian字典上的有效witness，但不是高阶N_k认证。精确物理基态由完整256维谱间隙与残差包围。', '',
        '能窗内振幅满足 |g0|≤1、|g4|≤sqrt(3/2)X、|g8|≤X²/4，因此重叠上界为 |beta0|+sqrt(3/2)|beta4|X+|beta8|X²/4。若此上界小于1，凸对偶不等式给出任何成本≤U的精确分解在该能窗内的系数质量比例上界。','',
        '默认U是明确给定的条件阈值；若另传入cost-certificates，同时使用该证书包围的实际Gaussian修补构造上界，此时存在成本≤U的精确表示。每行cost_threshold_kind区分两者。超过1的比例界按1截断并明确标为无信息。filled参考单列为source k=0，它不是replica优化结果。','',
        '| t | 方向来源k | 归一化k | Delta | U | L严格下界 | 能窗h严格上界 | 窗内质量占比上界 | 非平凡 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if abs(np.log10(r['delta'])+2)<1e-10:
            lines.append(f"| {r['t']:g} | {r['source_direction_order']} | 4 | {r['delta']:g} | {r['declared_cost_threshold']:g} | {r['L_physical_lower']:.7g} | {r['restricted_support_upper']:.7g} | {r['fraction_upper']:.7g} | {r['nontrivial']} |")
    lines+=['','完整21个窗口、归一化证书、物理基态误差与原始向量均在summary.json和NPZ。方向变化是否改进此保守证书由实际表值判断，不能从高阶标签预设。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
