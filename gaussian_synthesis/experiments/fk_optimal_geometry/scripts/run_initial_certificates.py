#!/usr/bin/env python3
"""A1 joint witnesses, B2 scalar stability, and D1/D2 directed bounds."""
from pathlib import Path
import argparse
import json
import hashlib
import sys
import time
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk5.paths import PREVIOUS
from fk5.io import source_manifest,write_json
from fk5.joint_dual_certificates import certify_joint
from fk5.restricted_fidelity_bounds import restricted_bound
from fk5.support_validation import spectral_certificate
from fk4.certificates import _bounds
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    costs=PREVIOUS/'results/column_cost_certificates_01'
    assert json.loads((costs/'verification.json').read_text())['artifact_verification_passed']
    previous=json.loads((costs/'summary.json').read_text())['rows'];model=build_model()
    joint=[];windows=[];spectra=[]
    for t in [5,9,10,10.9225,11,12,20]:
        U=next(r['corrected_Gaussian_cost_upper'] for r in previous if r['t']==t)
        row,arrays=certify_joint(model,t,U,Fraction(4713,1000) if t==20 else None)
        spectrum=spectral_certificate(model,t);spectra.append({'t':t,**spectrum})
        joint.append(row);np.savez_compressed(args.output/f't{t:g}_witness.npz',**arrays)
        print('joint',t,'L',row['joint_cost_lower'],'U',U,'gap',row['relative_gap_upper'],flush=True)
        if t in [9,11,20]:
            for delta in ['.001','.01','.1','1']:
                for multiplier in ['1','1.1','1.5','2']:
                    q=Fraction(multiplier);iv.dps=70
                    B=_bounds(iv.mpf(U)*q.numerator/q.denominator)[1]
                    # B is the explicit outward-rounded budget being certified.
                    result=restricted_bound(t,delta,B,row['L0_physical_interval'][0],spectrum['full_spectral_gap_lower'])
                    result['budget_multiplier']=multiplier;windows.append(result)
    assert next(r['relative_gap_upper'] for r in joint if r['t']==20)<.01
    write_json(args.output/'summary.json',{'status':'complete','A1_rows':joint,'D1_D2_rows':windows,'spectra':spectra,
        'A1_t20_gap_below_one_percent':True,'independent_verification_required':True,
        'whole_plan_complete':False,'cost_source':str(costs),'cost_source_sha256':hashlib.sha256((costs/'summary.json').read_bytes()).hexdigest(),
        'elapsed_seconds':time.monotonic()-started})
    lines=['# 初轮严格证书：A1、B2与D1/D2','',
        '此轮完成联合witness与标量组成稳定性、能窗fidelity及能量误差的区间计算。独立验证另行执行；不代表A–F整体完成。',
        '', '| t | 联合witness下界 | 已有严格U | 最强下界的相对缺口上界 | 必需平均eta区间 |',
        '|---:|---:|---:|---:|---|']
    for r in joint:lines.append(f"| {r['t']:g} | {r['joint_cost_lower']:.10g} | {r['cost_upper']:.10g} | {r['relative_gap_upper']:.6g} | {r['all_cost_le_U_mean_eta_interval']} |")
    lines+=['','| t | Delta | 预算约为U | 最低态距离 | fidelity上界 | 基态能量误差下界 |','|---:|---:|---:|---:|---:|---:|']
    for r in windows:
        if r['delta']==.01 and r['budget_multiplier']=='1':lines.append(f"| {r['t']:g} | .01 | {r['coefficient_budget']:.9g} | {r['phase_aligned_state_distance_lower']:.7g} | {r['fidelity_upper']:.7g} | {r['ground_energy_error_lower']:.7g} |")
    lines+=['','所有界针对完整Gaussian字典或指定能窗内的任意项数，要求近似态归一化且系数绝对值和受预算约束。局部支持搜索不参与这些证书。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
