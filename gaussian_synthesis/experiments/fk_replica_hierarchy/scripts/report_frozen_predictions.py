#!/usr/bin/env python3
"""Report measured frozen-dictionary transfer and full coherent kernels."""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t as student_t
from fk4.io import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    args=p.parse_args()
    s=json.loads((args.run/'summary.json').read_text())
    verification=json.loads((args.run/'verification.json').read_text())
    assert s['status']=='complete' and verification['artifact_verification_passed']
    os.environ['MPLCONFIGDIR']=str(args.run/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups={}
    for r in s['rows']:groups.setdefault((r['t'],r['epsilon'],r['method'],r['budget']),[]).append(r)
    aggregates=[]
    metrics=['fidelity','energy_error','mass_correlation_error','wick_deviation_error','retained_rank',
             'total_cold_prediction_seconds','prediction_seconds','dictionary_generation_seconds','dictionary_selection_seconds','source_generation_seconds_cold']
    for (t,eps,method,budget),rows in sorted(groups.items()):
        out={'t':t,'epsilon':eps,'method':method,'actual_terms':budget,'replicates':len(rows)}
        for field in metrics:
            v=np.array([r[field] for r in rows])
            out[field+'_mean']=float(v.mean())
            out[field+'_95_halfwidth']=float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else None
        for field in ['energy','mass_correlation','wick_deviation']:
            for kind in ['coherent','incoherent']:
                values=np.array([r[kind][field] for r in rows])
                out[kind+'_'+field+'_mean']=float(values.mean())
                out[kind+'_'+field+'_95_halfwidth']=float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        out['mass_kernel_off_diagonal_mean']=float(np.mean([r['coherent_kernel_parts']['mass_correlation']['off_diagonal'] for r in rows]))
        aggregates.append(out)
    write_json(args.run/'aggregates.json',aggregates)
    with (args.run/'aggregates.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
    names={'random':'Random','physics_prior':'Physical prior','dual_columns':'Dual columns',
           'husimi_train_frozen':'Training local maxima','husimi_localmax_test_oracle':'Test local maxima (oracle)',
           'husimi_test_oracle':'Test Husimi distribution (oracle)','single_gaussian_HFB':'Single Gaussian HFB',
           'replica_k2_P':'Replica k=2 P'}
    methods=sorted({r['method'] for r in s['rows']})
    colors={'random':'#b0b0b0','physics_prior':'#2ca02c','dual_columns':'#d62728',
            'single_gaussian_HFB':'#555555','husimi_test_oracle':'#9467bd',
            'husimi_localmax_test_oracle':'#c49c94','husimi_train_frozen':'#8c564b',
            'replica_k2_P':'#1f77b4','replica_k8_P':'#ff7f0e','replica_k16_P':'#17becf','replica_k32_P':'#e377c2'}
    ts=sorted({r['t'] for r in s['rows']})
    for k in [2,8,16,32]:names[f'replica_k{k}_P']=f'Replica k={k} P'
    complete_replica_matrix=all((t,eps,f'replica_k{k}_P',budget) in groups and len(groups[t,eps,f'replica_k{k}_P',budget])==5
        for t in [9,10,10.9225,11,12,20] for eps in [0,.02,.05,.1,.2] for k in [2,8,16,32] for budget in [1,2,4,8,16,32,64])
    write_json(args.run/'comparison_coverage.json',{'complete_required_replica_matrix':complete_replica_matrix,
        'prediction_rows':len(s['rows']),'all_methods':methods,'physical_points':ts,
        'all_frozen_nonoracle_predictions_precede_ED':all(t['all_nonoracle_predictions_completed_before_ED'] for t in s['targets'])})
    (args.run/'figures').mkdir(exist_ok=True)
    for metric,label,filename in [('fidelity','Ground-state fidelity','transfer_fidelity'),('energy_error','Variational energy error','transfer_energy')]:
        energy_plot_floor=min(r['energy_error_mean'] for r in aggregates if r['epsilon']==.2)/3
        columns=min(3,len(ts));row_count=(len(ts)+columns-1)//columns
        fig,axes=plt.subplots(row_count,columns,figsize=(4.4*columns,3.9*row_count+.4),sharex=True,squeeze=False)
        for ax,t in zip(axes.flat,ts):
            for method in methods:
                rows=[r for r in aggregates if r['t']==t and r['epsilon']==.2 and r['method']==method]
                if not rows:continue
                rows.sort(key=lambda x:x['actual_terms'])
                values=[max(1e-13,r[metric+'_mean']) for r in rows]
                errors=[r[metric+'_95_halfwidth'] for r in rows]
                x=[r['actual_terms'] for r in rows]
                if metric=='energy_error':
                    # Symmetric repeat intervals can cross the physical bound 0.
                    # Keep the upper interval; visibly mark the truncated lower end.
                    plotted_errors=np.array([[min(e,max(v-energy_plot_floor,0)) for v,e in zip(values,errors)],errors])
                    clipped=[xx for xx,v,e in zip(x,values,errors) if v-e<energy_plot_floor]
                    if clipped:ax.scatter(clipped,[energy_plot_floor]*len(clipped),marker='v',s=22,color=colors.get(method),zorder=4)
                else:plotted_errors=errors
                ax.errorbar(x,values,yerr=plotted_errors,marker='o',markersize=3,
                            capsize=2,linewidth=1.2,label=names.get(method,method),color=colors.get(method),linestyle='--' if 'oracle' in method else '-')
            ax.set(title=f't={t:g}, epsilon=0.2',xscale='log',xlabel='Actual Gaussian terms',ylabel=label)
            ax.set_xticks([1,2,4,8,16,32,64],labels=['1','2','4','8','16','32','64'])
            ax.grid(alpha=.2)
            if metric=='fidelity':ax.set_ylim(-.03,1.03)
            else:
                ax.set_yscale('log');ax.set_ylim(bottom=energy_plot_floor*.75)
        for ax in list(axes.flat)[len(ts):]:ax.set_visible(False)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=8)
        if metric=='energy_error':fig.text(.5,.005,'Down triangles: lower repeat interval extends below the displayed positive floor.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.025 if metric=='energy_error' else 0,1,.9))
        for suffix in ['png','pdf']:fig.savefig(args.run/'figures'/f'{filename}.{suffix}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
    selected=[m for m in ['replica_k2_P','replica_k8_P','replica_k16_P','replica_k32_P','physics_prior','dual_columns','single_gaussian_HFB'] if m in methods]
    targets=sorted([r for r in s['targets'] if r['epsilon']==0],key=lambda r:r['t'])
    for ax,kind in zip(axes,['coherent','incoherent']):
        ax.plot(range(len(ts)),[r['metrics']['wick_deviation'] for r in targets],'k-',linewidth=2,label='Exact target')
        for method in selected:
            selected_rows=sorted([r for r in aggregates if r['epsilon']==0 and r['method']==method and
                                  (r['actual_terms']==32 or method=='single_gaussian_HFB')],key=lambda r:r['t'])
            ax.errorbar([ts.index(r['t']) for r in selected_rows],[r[kind+'_wick_deviation_mean'] for r in selected_rows],
                        yerr=[r[kind+'_wick_deviation_95_halfwidth'] for r in selected_rows],marker='o',markersize=3,capsize=2,label=names[method],color=colors[method])
        ax.set(title=kind.capitalize()+' composition',xlabel='t (listed parameter points)',ylabel='Mass-correlation Wick deviation')
        ax.set_xticks(range(len(ts)),[f'{t:g}' for t in ts],rotation=25)
        ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.87))
    for ext in ['png','pdf']:fig.savefig(args.run/'figures'/f'coherent_wick_comparison.{ext}',dpi=180)
    plt.close(fig)
    lines=['# 冻结 Gaussian 字典：相干核与非均匀扰动对照','',
           f"已完成并独立核验 {len(s['rows'])} 个预测，覆盖 {len(ts)} 个 t、5 个扰动强度、5 个随机重复。当前 replica 字典为 {', '.join(m for m in methods if m.startswith('replica_'))}。"+('方案要求的六点、四阶、七种实际项数预算对照已齐。' if complete_replica_matrix else '高阶对照未齐，不将本轮视为完整E4/E5结论。'),'',
           '所有普通字典在新 Hamiltonian 的 ED 之前冻结；系数由 H/S 广义本征问题得到。ED 随后计算验证指标。两类 test oracle 明确使用测试目标：局部最大化与精确 Haar 拒绝采样的 Husimi 分布，不能称为预测方法。', '',
           '计数是实际完整 Gaussian ket 数目；没有把 Spin(7) 轨道代表当作一个 Gaussian 项。HFB 只报告1项，采用32个起点的完整Gaussian流形局部优化；未认证为全局最优。局部最大化字典可能退化，表中同时报告保留秩。误差线是五个随机重复的经验 Student-t 95% 半宽。','',
           '## 扰动迁移','',
           '![保真度](figures/transfer_fidelity.png)','',
           '![能量误差](figures/transfer_energy.png)','',
           '能量图使用对数轴；若对称重复区间的下端低于显示范围，以底部向下三角标记，未把负的区间下端解释为负变分误差。完整未截断重复区间保存在CSV。','',
           '以下为 epsilon=0.2、32项（HFB为1项）的均值。完整所有预算/扰动/误差线在 aggregates.csv。','',
           '| t | 方法 | 实际项数 | 保留秩 | 保真度 | 能量误差 | 质量关联误差 | Wick偏离误差 |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        if r['epsilon']==.2 and (r['actual_terms']==32 or r['method']=='single_gaussian_HFB'):
            lines.append(f"| {r['t']:g} | {names.get(r['method'],r['method'])} | {r['actual_terms']} | {r['retained_rank_mean']:.3g} | {r['fidelity_mean']:.7g} | {r['energy_error_mean']:.5g} | {r['mass_correlation_error_mean']:.5g} | {r['wick_deviation_error_mean']:.5g} |")
    lines+=['','## 相干与非相干组成','',
            '![相干与非相干Wick比较](figures/coherent_wick_comparison.png)','',
            '以下为 epsilon=0、32项（HFB为1项）。非相干混合使用同一组原子和 |c_i|² 归一化权重；其 Wick 预测从混合协方差重新计算。未另行优化混合权重，因此该对照不代表所有非相干Gaussian混合的最优界。相干矩阵的对角部分和该归一化混合不是同一量。所有 H、质量关联和三个四点算符的完整核保存在逐预测 NPZ。','',
            '| t | 方法 | 相干Wick偏离 | 非相干Wick偏离 | 质量关联非对角贡献 |',
            '|---:|---|---:|---:|---:|']
    for r in aggregates:
        if r['epsilon']==0 and (r['actual_terms']==32 or r['method']=='single_gaussian_HFB'):
            lines.append(f"| {r['t']:g} | {names.get(r['method'],r['method'])} | {r['coherent_wick_deviation_mean']:.6g} | {r['incoherent_wick_deviation_mean']:.6g} | {r['mass_kernel_off_diagonal_mean']:.6g} |")
    lines+=['','## 成本与证据范围','',
            'aggregates.csv 分别给出来源计算、候选生成、选择、预测和总冷启动秒数。上游 witness/采样/列生成整轮可复用成本均计入，未按字典数量均摊；因此总冷启动列是保守共享成本口径，不是相同时限竞赛。GPU批次与CPU批次的时间也不能直接当作硬件无关复杂度。','',
            '独立核验重新构造完整256维 Hamiltonian、所有相干核、完整 Majorana 协方差和三项 Wick 收缩；用SVD子空间本征值交叉检查广义本征求解，并核验冻结原子哈希、真实项数、Gaussian纯度与成本求和。','',
            '这些观测量比较刻画给定有限字典和预算的限制；不会仅凭Wick峰附近的数值变化声称已证明高阶接触集合突变或唯一必需构型。']
    (args.run/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
