#!/usr/bin/env python3
"""Figure 1 from saved observations, retaining incomplete/failed-gate marks."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4.io import write_json,source_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--columns',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',type=Path,help='Verified exact-direction t0 scan; supplements orders absent from the ordinary runs')
    p.add_argument('--block-certificates',type=Path,nargs='*',default=[])
    p.add_argument('--final',action='store_true',help='Require the complete declared E1 grid and label this as the final scan')
    p.add_argument('--cost-certificates',type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    best=json.loads((args.columns/'best_costs.json').read_text())['rows']
    certified_costs={}
    if args.cost_certificates:
        assert json.loads((args.cost_certificates/'verification.json').read_text())['artifact_verification_passed']
        certified_costs={r['t']:r['corrected_Gaussian_cost_upper'] for r in json.loads((args.cost_certificates/'summary.json').read_text())['rows']}
    rows=[]
    for run in args.runs:
        s=json.loads((run/'summary.json').read_text())
        assert s['status']=='complete'
        assert json.loads((run/'verification.json').read_text())['artifact_verification_passed']
        for r in s['rows']:
            exact='independent_full_rho_upper' in r
            cost=next(b for b in best if b['t']==r['t'])
            Ubest=2. if r['t']==0 else certified_costs.get(r['t'],cost['U'])
            L=r['L_certified_saved_target'] if exact else r['L_point_mean']
            rho=r['independent_full_rho_upper'] if exact else r['rho_RB_pooled']
            passed=r['mass_repeat_pass'] and (rho<1e-3 if exact else r['all_seeds_reconstruction_pass'])
            U=min(Ubest,r['B_mean'])
            rows.append({'t':r['t'],'k':r['k'],'L':L,'L_certified_saved_target':exact,
                         'B':r['B_mean'],'B_95_halfwidth':r['B_95_percent_halfwidth'],
                         'B_is_strict_upper':False,'U_best':Ubest,'U_best_strictly_certified':r['t']==0 or r['t'] in certified_costs,
                         'U_best_original_floating_cost':cost['U'],
                         'rho':rho,'rho_kind':'interval full-Fock upper' if exact else 'pooled Monte Carlo RB residual',
                         'gate_pass':passed,'G_point':(U-L)/U,'G_point_is_certified':False,
                         'source_run':str(run.resolve()),'P_saved':True})
    if args.endpoint:
        endpoint=json.loads((args.endpoint/'summary.json').read_text())
        assert json.loads((args.endpoint/'verification.json').read_text())['artifact_verification_passed']
        existing={(r['t'],r['k']) for r in rows}
        for r in endpoint['rows']:
            if (0.,r['k']) in existing:
                continue
            keys={o['key'] for o in r['observations']}
            verification=json.loads((args.endpoint/'verification.json').read_text())
            rho=max(o['all_branch_reconstruction_residual'] for o in verification['observations'] if o['key'] in keys)
            L=r['L_canonical_point_mean']
            rows.append({'t':0.,'k':r['k'],'L':L,'L_certified_saved_target':r['k']==1,
                         'B':r['B_mean'],'B_95_halfwidth':r['B_95_percent_halfwidth'],
                         'B_is_strict_upper':False,'U_best':2.,'U_best_strictly_certified':True,
                         'rho':rho,'rho_kind':'complete 256N-branch finite realization; exact canonical direction proved',
                         'gate_pass':r['mass_repeat_pass'],'G_point':(2-L)/2,'G_point_is_certified':r['k']==1,
                         'source_run':str(args.endpoint.resolve()),'P_saved':True,
                         'L_support_alternative_exact':2.,'actual_finite_Gaussian_terms':r['actual_branches_per_empirical_decomposition']})
    for directory in args.block_certificates:
        block=json.loads((directory/'summary.json').read_text())
        assert json.loads((directory/'verification.json').read_text())['artifact_verification_passed']
        for c in block['rows']:
            for r in rows:
                if (r['t'],r['k'])==(c['t'],c['direction_order']):
                    r['L_conservative_block_certificate']=c['L_certified_exact_physical_target']
                    r['Nk_certificate_anchor_order']=c['anchor_order']
    if args.final:
        required={0:[1,2,4,8,16,32,64,128],5:[2,4,8,16],9:[1,2,4,8,16,32],
                  10:[1,2,4,8,16,32],10.9225:[2,4,8,16,32],11:[1,2,4,8,16,32],
                  12:[2,4,8,16],20:[1,2,4,8,16,32,64]}
        present={(r['t'],r['k']) for r in rows}
        assert len(present)==len(rows),'duplicate physical/order points'
        assert all((t,k) in present for t,ks in required.items() for k in ks),'required scan incomplete'
    os.environ['MPLCONFIGDIR']=str(args.output/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ts=sorted({r['t'] for r in rows})
    fig,axes=plt.subplots(3,3,figsize=(13,11),sharex=True)
    for ax,t in zip(axes.flat,ts):
        group=sorted([r for r in rows if r['t']==t],key=lambda r:r['k'])
        k=[r['k'] for r in group]
        ax.plot(k,[r['B'] for r in group],color='C1',alpha=.4)
        for r in group:
            ax.errorbar(r['k'],r['B'],yerr=r['B_95_halfwidth'],fmt='o' if r['gate_pass'] else 'x',
                        color='C1' if r['gate_pass'] else 'crimson',capsize=3)
            ax.plot(r['k'],r['L'],'o',color='C0',markerfacecolor='C0' if r['L_certified_saved_target'] else 'none')
            if 'L_conservative_block_certificate' in r:
                ax.plot(r['k'],r['L_conservative_block_certificate'],'^',color='C2')
        ax.axhline(group[0]['U_best'],color='black',linestyle='-' if group[0]['U_best_strictly_certified'] else '--',linewidth=1)
        ax.set(xscale='log',title=f't={t:g}',xlabel='Replica order k',ylabel='Gaussian coefficient cost')
        ax.set_xticks([1,2,4,8,16,32,64,128],[1,2,4,8,16,32,64,128])
        ax.set_xlim(.8,145)
        ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker='o',color='C0',linestyle='none',label='L: certified saved target'),
                        Line2D([],[],marker='o',color='C0',markerfacecolor='none',linestyle='none',label='L: uncertified estimate'),
                        Line2D([],[],marker='^',color='C2',linestyle='none',label='L: conservative exact-block certificate'),
                        Line2D([],[],marker='o',color='C1',linestyle='none',label='B: empirical 95% repeat interval'),
                        Line2D([],[],marker='x',color='crimson',linestyle='none',label='B: reconstruction gate failed'),
                        Line2D([],[],color='black',linestyle='-' if certified_costs else '--',label='Independent U: certified Gaussian repair' if certified_costs else 'Independent finite U (floating arithmetic)')],
               loc='lower center',ncol=2,fontsize=9)
    fig.suptitle('Replica hierarchy: complete declared scan; certification shown separately' if args.final else 'Replica hierarchy: measured orders only; higher orders pending',fontsize=13)
    fig.tight_layout(rect=(0,.09,1,.96))
    for ext in ['png','pdf']:
        fig.savefig(args.output/f'figure1_progress.{ext}',dpi=180)
    write_json(args.output/'observations.json',{'rows':rows,'complete_declared_E1_scan':args.final,
               'all_reconstruction_and_mass_gates_pass':all(r['gate_pass'] for r in rows),
               'high_order_canonical_L_certified':False})
    lines=['# Figure 1：Replica hierarchy 扫描' if args.final else '# Figure 1：Replica hierarchy 进度图','',
           '仅绘制已测量的数据；分级扫描没有要求所有参数与阶数的笛卡尔积。红叉表示重构门槛未通过。k=1、2、4 的 L 和完整态残差已用解析/精确矩验证；k≥8 的 canonical L 仍是未认证的积分点估计。绿色三角单独表示有精确Nk上界的保守witness，不能与canonical归一化混同。B 保留五次独立重复的经验不确定度。',
           '', '![Replica hierarchy](figure1_progress.png)','',
           '| t | k | L | B ± 95%半宽 | U_best | rho | 验收 |',
           '|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(rows,key=lambda r:(r['t'],r['k'])):
        lines.append(f"| {r['t']:g} | {r['k']} | {r['L']:.7g}{'' if r['L_certified_saved_target'] else '（未认证）'} | {r['B']:.7g} ± {r['B_95_halfwidth']:.2g} | {r['U_best']:.7g} | {r['rho']:.3g} | {r['gate_pass']} |")
    lines+=['','U_best 在 t=0 采用精确解析值2。若传入cost-certificates，其他点采用相邻精确Gaussian分支与完整Fock修补的保守成本证书（IEEE舍入假设明确列出），原浮点成本另列保留；否则仍是浮点构造成本。canonical L的高阶点估计尚未认证，因此G_point依然不是严格缺口。',
            '', '保存目标态到精确 Hamiltonian 基态的距离已有独立上界（<1.4e-14），可用于转移低阶 L；对应独立证书保存在 cartan_k2_physical_certificates_01 和 cartan_k4_physical_certificates_02。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
