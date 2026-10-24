#!/usr/bin/env python3
"""Verify saved finite Gaussian expansions and summarize independent costs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import write_json
from fk4.distributions import full_distribution
from fk.model import build_model
from fk.observables import atom_invariants


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    args=p.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    assert summary['status']=='complete'
    manifest=json.loads((args.run/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((args.run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    model=build_model()
    total=0
    checked=[]
    for row in summary['prior_rows']+summary['rows']:
        data=np.load(args.run/f"{row['key']}.npz")
        finite=row.get('finite_expansion',row)
        g,c=data['finite_kets'],data['finite_coefficients']
        residual=data['target']-c@g
        np.testing.assert_allclose(data['fock_repair'],residual,atol=2e-12)
        assert len(g)==finite['main_gaussian_branch_count']
        np.testing.assert_allclose(abs(c).sum()+abs(residual).sum(),finite['U_finite_repaired'],atol=2e-12)
        np.testing.assert_allclose(np.linalg.norm(g,axis=1),1,atol=1e-10,rtol=0)
        features,diagnostic=full_distribution(g,model.j_matrices)
        saved=atom_invariants(data['finite_covariances'],model.j_matrices)
        expected=np.column_stack([saved[k] for k in ['mass_a','mass_b','mass_dot','eta']])
        np.testing.assert_allclose(features['features'],expected,atol=2e-11)
        projected=model.S@(model.S.conj().T@data['atoms'].T)@data['coefficients']
        orbit_cost=abs(data['coefficients']).sum()+abs(data['target']-projected).sum()
        np.testing.assert_allclose(orbit_cost,row['U_orbit_repaired'],atol=2e-11)
        total+=len(g)
        checked.append({'key':row['key'],'actual_main_branches':len(g),'purity_error':diagnostic['max_purity_error_all_particles']})
        print('verified',row['key'],len(g),flush=True)
    best=[]
    for t in sorted({r['t'] for r in summary['rows']}):
        candidates=[]
        for r in summary['prior_rows']+summary['rows']:
            if r['t']==t:
                f=r.get('finite_expansion',r)
                candidates.append({'U':f['U_finite_repaired'],'key':r['key'],
                                   'actual_main_branches':f['main_gaussian_branch_count'],
                                   'fock_repair_branches':f['fock_repair_count_nonzero']})
        chosen=min(candidates,key=lambda r:r['U'])
        best.append({'t':t,**chosen,'strict_interval_certified':False,
                     'analytic_cost_upper_if_t0':2. if t==0 else None})
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,'checked_expansions':len(checked),
                'all_actual_gaussian_branches_checked':total,'rows':checked,'strict_interval_cost_certified':False,
                'global_optimality_proved':False})
    write_json(args.run/'best_costs.json',{'status':'complete','rows':best})
    lines=['# 独立列生成与有限 Gaussian 构造','',
           '每个参数使用 5 个独立随机种子。局部 pricing 搜索不构成全局支持函数上界；下表不宣称成本已最优。所有构造在完整 Fock 空间中加入残差修复，并按实际 Gaussian 分支数计数。',
           '', '| t | 最好有限构造成本 | 主 Gaussian 分支 | Fock 修复分支 | 来源 |',
           '|---:|---:|---:|---:|---|']
    for r in best:
        lines.append(f"| {r['t']:g} | {r['U']:.10g} | {r['actual_main_branches']} | {r['fock_repair_branches']} | {r['key']} |")
    lines+=['',f'独立检查了 {len(checked)} 个有限分解、{total} 个主 Gaussian 分支。原子归一化、纯度、系数和全 Fock 重构通过浮点容差检查；有限构造成本尚未采用区间运算包围。',
            '', 't=0 另有精确解析成本 2，以及 PRODUCT_MAGIC_NECESSITY.md 中针对所有低成本分解的组成约束。']
    (args.run/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
