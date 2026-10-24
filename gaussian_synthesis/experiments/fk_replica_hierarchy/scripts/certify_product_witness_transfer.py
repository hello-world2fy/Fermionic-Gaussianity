#!/usr/bin/env python3
"""Transfer the exact w0 support theorem to all targets, including window bounds."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk4.io import write_json,source_manifest
from fk4.certificates import _c,_bounds,exact_basis_action
from fk4.state_enclosure import enclose_ground_state
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--costs',type=Path,default=Path('results/column_cost_certificates_01'));args=p.parse_args()
    assert json.loads((args.costs/'verification.json').read_text())['artifact_verification_passed']
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    costs=json.loads((args.costs/'summary.json').read_text())['rows']
    q4=json.loads(Path('results/cartan_k4_physical_certificates_02/summary.json').read_text())['rows']
    model=build_model();rows=[];windows=[]
    for source in costs:
        t=source['t'];target=model.target(t);enclosure=enclose_ground_state(model,t,target)
        iv.dps=70;w=exact_basis_action(np.array([2.,0.,0.],complex))
        inner=sum((a*_c(v)[0]+b*_c(v)[1] for (a,b),v in zip(w,target)),iv.mpf(0))
        L=_bounds(inner-2*iv.mpf(enclosure['saved_target_to_exact_ground_state_distance_upper']))[0]
        assert L>0
        U=source['corrected_Gaussian_cost_upper']
        if t==0:L=U=2.
        mean_eta=0. if t==0 else min(1.,_bounds(2*(1-iv.mpf(L)/iv.mpf(U)))[1])
        L4=next(r['L_certified_exact_ground_state'] for r in q4 if r['t']==t)
        best_lower=max(1.,L,L4)
        row={'t':t,'L_endpoint_witness_physical_lower':L,'U_certified':U,'mean_eta_upper_all_cost_le_U':mean_eta,
             'best_combined_physical_cost_lower':best_lower,
             'Gaussian_extent_lower':max(1.,_bounds(iv.mpf(best_lower)**2)[0]),
             'Gaussian_extent_upper':_bounds(iv.mpf(U)**2)[1],
             'certified_global_relative_gap_upper':max(0.,_bounds(1-iv.mpf(best_lower)/iv.mpf(U))[1]),
             'physical_target_enclosure':enclosure,
             'tail_bounds':[{'eta0':e,'fraction_upper':min(1.,_bounds(iv.mpf(mean_eta)/iv.mpf(e))[1])} for e in [.1,.2,.5,.8]],
             'support_proof':'PRODUCT_MAGIC_NECESSITY.md, |<2s0|g>|<=1-eta/2',
             'cost_source':str(args.costs.resolve())}
        if t==0:row['Gaussian_extent_lower']=row['Gaussian_extent_upper']=4.;row['certified_global_relative_gap_upper']=0.
        rows.append(row);np.savez_compressed(args.output/f't{t:g}.npz',target=target)
        if t not in [9,9.1,11,20]:continue
        for delta in np.logspace(-5,0,21):
            a=4*(iv.mpf(float(t))-9);d=iv.mpf(float(delta));X=2*d/(iv.sqrt(a*a+72*d)+a)
            gap=max(0.,_bounds(1-X)[0]);eta_lower=max(0.,_bounds(iv.mpf(gap)**2)[0])
            fraction=1. if eta_lower==0 else min(1.,_bounds(iv.mpf(mean_eta)/iv.mpf(eta_lower))[1])
            windows.append({'t':t,'delta':float(delta),'X_interval':_bounds(X),'eta_window_lower':eta_lower,
                'restricted_support_upper':_bounds(1-iv.mpf(eta_lower)/2)[1],
                'fraction_upper':fraction,'L_endpoint_witness_physical_lower':L,'U_certified':U,
                'source':'exact endpoint support witness, not a replica direction or q4 normalization'})
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'window_rows':windows,
        'cost_certificate_sha256':hashlib.sha256((args.costs/'summary.json').read_bytes()).hexdigest(),
        'endpoint_witness_independent_of_target':True,'comparison_is_not_a_high_order_integral_result':True})
    lines=['# 解析端点witness的跨参数必要性基线','',
        '使用w0=2M_AM_B的完整Gaussian尖锐支持界，直接约束全部参数点的低成本分解。U是独立构造的严格上界，L0转移到已包围的精确物理基态。此基线必须与高阶witness公平比较。证明见PRODUCT_WITNESS_TRANSFER.md。','',
        '| t | L0严格下界 | 独立U | 最强已认证成本下界 | 严格相对缺口上界 | 所有B≤U分解的平均eta上界 |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['t']:g} | {r['L_endpoint_witness_physical_lower']:.9g} | {r['U_certified']:.9g} | {r['best_combined_physical_cost_lower']:.9g} | {r['certified_global_relative_gap_upper']:.6g} | {r['mean_eta_upper_all_cost_le_U']:.6g} |")
    lines+=['','| t | Delta | 能窗内Gaussian的eta下界 | 任意B≤U分解的窗内系数质量比例上界 |','|---:|---:|---:|---:|']
    for r in windows:
        if abs(r['delta']-.01)<1e-15:lines.append(f"| {r['t']:g} | .01 | {r['eta_window_lower']:.9g} | {r['fraction_upper']:.9g} |")
    lines+=['','这些解析参考界在本轮窗口上强于保守q4归一化的高阶方向。因此，虽然高阶方向改善了同类q4参考界，却不能声称它们优于所有已知物理witness。eta不等式使用L0，不能替换成别的witness的更强L。Gaussian extent界是系数成本界的平方，另存summary.json。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
