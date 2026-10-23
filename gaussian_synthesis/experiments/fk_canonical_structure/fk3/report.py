"""Generate a research report from frozen arrays, without new experiments."""
import json
import os
from pathlib import Path
import numpy as np


def _read(path,default):
    return json.loads(path.read_text()) if path.exists() else default


def make_report(output):
    output=Path(output).resolve()
    os.environ['MPLCONFIGDIR']=str(output/'.mplconfig')
    os.environ['XDG_CACHE_HOME']=str(output/'.cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    config=_read(output/'config.json',{})
    standard_config=_read(Path(__file__).resolve().parents[1]/'configs/canonical.json',{})
    development_diagnostic=(not standard_config or config != standard_config)
    seed_count=len(config['seeds'])
    atom_count=config['discovery']['atom_count']
    candidate_count=len(config['t_values'])+len(config['k4_t_values'])
    failure_probability=config['total_failure_probability']
    event_count=config['planned_statistical_events']
    summary=_read(output/'summary.json',{})
    exact=_read(output/'canonical_k2.json',{'rows':[]})
    acceptance=_read(output/'acceptance.json',{'rows':[]})
    gates={r['id']:r for r in acceptance['rows']}
    validation=[]; discoveries=[]; stores={}
    for seed in config['seeds']:
        data=_read(output/f'validation_{seed}.json',{'rows':[]})
        validation.extend([dict(r,seed=seed) for r in data['rows'] if r['n']==config['final_samples']])
        discoveries.extend([dict(r,seed=seed) for r in _read(output/f'discovery_{seed}.json',{'rows':[]})['rows']])
        if (output/f'validation_{seed}.npz').exists():
            with np.load(output/f'validation_{seed}.npz') as a:
                stores[seed]={k:a[k] for k in a.files}
    canonical=[r for r in validation if not r['is_probe']]
    figdir=output/'figures'; figdir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(figdir/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    def values(rows,field):
        return np.asarray([r[field] for r in rows],float)
    def num(x):
        return '—' if x is None else f'{x:.6g}'
    def median(rows,field):
        return float(np.median(values(rows,field))) if rows else None
    def extent(rows,field):
        a=values(rows,field)
        return f'{a.min():.4g}–{a.max():.4g}' if len(a) else '—'

    if canonical:
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for k,color in [(2,'#0d766e'),(4,'#b45309')]:
            ts=sorted({r['t'] for r in canonical if r['k']==k})
            for field,style,label in [('raw_residual','--','raw Haar'),('rb_residual','-','orbit average')]:
                med=[]; lo=[]; hi=[]
                for t in ts:
                    a=values([r for r in canonical if r['k']==k and r['t']==t],field)
                    med.append(np.median(a)); lo.append(a.min()); hi.append(a.max())
                axs[0].plot(ts,med,style,marker='o',color=color,label=f'k={k}, {label}')
                axs[0].fill_between(ts,lo,hi,color=color,alpha=.08)
            xs=[]; ys=[]; lows=[]; highs=[]
            for t in ts:
                group=[r for r in canonical if r['k']==k and r['t']==t]
                gate=gates.get(group[0]['id'],{})
                y=median(group,'mean_eta'); a=values(group,'mean_eta')
                axs[1].errorbar(t,y,yerr=[[y-a.min()],[a.max()-y]],fmt='o',color=color,
                    markerfacecolor=color if gate.get('distribution_interpretation_allowed') else 'white',capsize=3)
                xs.append(t); ys.append(y)
            axs[1].plot(xs,ys,color=color,alpha=.6,label=f'k={k} (lines guide only)')
        axs[0].axhline(config['reconstruction_tolerance'],color='black',ls=':',label='MC precision gate')
        axs[0].set(xlabel='t / V',ylabel='Held-out reconstruction residual',yscale='log')
        axs[0].legend(fontsize=8)
        axs[1].set(xlabel='t / V',ylabel='Canonical coefficient-mass mean eta',
                   title='Filled: marginal-distribution gates pass')
        axs[1].legend(fontsize=8)
        save(fig,'canonical_diagnostics')

    cost_summary=[]
    for t in config['t_values']:
        groups={m:[r for r in discoveries if r['t']==t and r['method']==m] for m in ('random','guided','physics_arc')}
        entry={'t':t,'complete':all(len(v)==len(config['seeds']) for v in groups.values())}
        for method,group in groups.items():
            entry[method+'_median']=median(group,'cost_upper')
        paired=[]
        for seed in config['seeds']:
            p={m:next((r for r in group if r['seed']==seed),None) for m,group in groups.items()}
            if all(p.values()):
                paired.append({'seed':seed,'guided_to_random':p['guided']['cost_upper']/p['random']['cost_upper'],
                               'guided_to_prior':p['guided']['cost_upper']/p['physics_arc']['cost_upper'],
                               'guided_within_5pct_prior':p['guided']['cost_upper']<=1.05*p['physics_arc']['cost_upper'],
                               'guided_strictly_below_random':p['guided']['cost_upper']<p['random']['cost_upper']-1e-8})
        entry['paired']=paired
        cost_summary.append(entry)
    if discoveries:
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for method,color in [('random','#64748b'),('guided','#0d766e'),('physics_arc','#b45309')]:
            xs=[]; ys=[]; lo=[]; hi=[]
            for t in config['t_values']:
                group=[r for r in discoveries if r['t']==t and r['method']==method]
                if not group: continue
                a=values(group,'cost_upper');xs.append(t);ys.append(np.median(a));lo.append(a.min());hi.append(a.max())
                axs[1].scatter([t]*len(group),values(group,'coefficient_mass_mean_eta'),color=color,s=20,alpha=.65)
            axs[0].plot(xs,ys,'o-',color=color,label=method)
            axs[0].fill_between(xs,lo,hi,color=color,alpha=.12)
        axs[0].set(xlabel='t / V',ylabel='Repaired orbit representation upper U',title=f'128 Fock + {atom_count} new atoms per arm')
        axs[0].legend()
        axs[1].set(xlabel='t / V',ylabel='Absolute-coefficient-mass mean eta',title='SOCP orbit coefficient mass')
        save(fig,'discovery_costs')

    allowed=[r for r in acceptance['rows'] if r['distribution_interpretation_allowed']]
    histogram_masks={}
    haar_masks={}
    if allowed and stores:
        fig,axs=plt.subplots((len(allowed)+1)//2,4,figsize=(15,3.7*((len(allowed)+1)//2)),squeeze=False)
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#d1d5db')
        for position,gate in enumerate(allowed):
            hist=[]; ess=[]; haar=[]
            for seed,a in stores.items():
                j=list(a['candidate_ids']).index(gate['id'])
                hist.append(a[f'n{config["final_samples"]}_histogram'][j])
                ess.append(a[f'n{config["final_samples"]}_histogram_ess'][j])
                haar.append(a[f'n{config["final_samples"]}_haar_histogram'])
            mask=np.min(ess,axis=0)>=config['bin_ess_min']
            # Haar reference has equal weights: local ESS is the bin count,
            # not the canonical coefficient-weighted ESS.
            haar_mask=np.min(haar,axis=0)*config['final_samples']>=config['bin_ess_min']
            histogram_masks[gate['id']]=mask.tolist()
            haar_masks[gate['id']]=haar_mask.tolist()
            for offset,(data,valid,label) in enumerate(((hist,mask,'canonical'),(haar,haar_mask,'Haar reference'))):
                ax=axs.ravel()[2*position+offset]
                plotted=np.ma.array(np.median(data,axis=0),mask=~valid)
                im=ax.imshow(plotted,origin='lower',extent=[0,1,0,1],vmin=0,vmax=1,cmap=cmap)
                ax.set(xlabel='eta',ylabel='local mass_A squared',
                       title=f't={gate["t"]:g}, k={gate["k"]}: {label}')
                fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='mass fraction / bin')
        for ax in axs.ravel()[2*len(allowed):]:ax.set_visible(False)
        save(fig,'accepted_marginals')

    coherent=[r for r in acceptance['rows'] if r['coherent_interpretation_allowed']]
    kernel_ranges={}
    for gate in acceptance['rows']:
        kernels=[]; kernel_seeds=[]
        for seed,a in stores.items():
            if gate['id'] not in a['candidate_ids']:
                continue
            j=list(a['candidate_ids']).index(gate['id'])
            kernels.append(a[f'n{config["final_samples"]}_K_cross'][j])
            kernel_seeds.append(seed)
        if not kernels:
            continue
        kernels=np.asarray(kernels)
        complete=set(kernel_seeds)==set(config['seeds'])
        row={'seeds':kernel_seeds,'complete_configured_seed_set':complete,
             'interpretation_allowed':bool(gate['coherent_interpretation_allowed'] and complete),
             'cross_seed_stability_assessable':bool(complete and len(kernels)>=2),
             'ranges_are_confidence_intervals':False}
        for name,part in (('real',kernels.real),('imag',kernels.imag)):
            minimum=part.min(axis=0);maximum=part.max(axis=0)
            row.update({name+'_minimum':minimum.tolist(),name+'_maximum':maximum.tolist(),
                        name+'_range':(maximum-minimum).tolist(),name+'_median':np.median(part,axis=0).tolist()})
        row['real_sign_consistent_across_seeds']=(
            (((kernels.real.min(axis=0)>0)|(kernels.real.max(axis=0)<0))
             & row['cross_seed_stability_assessable']).tolist())
        kernel_ranges[gate['id']]=row
    if coherent and stores:
        fig,axs=plt.subplots(len(coherent),2,figsize=(9,3.8*len(coherent)),squeeze=False)
        for axes,gate in zip(axs,coherent):
            stats=kernel_ranges[gate['id']]
            matrix=np.asarray(stats['real_median']);spread=np.asarray(stats['real_range'])
            lim=max(1e-9,np.max(abs(matrix)))
            stable=np.asarray(stats['real_sign_consistent_across_seeds'])
            for ax,data,label,cmap_name,vmin,vmax in (
                (axes[0],matrix,'median Re K','RdBu_r',-lim,lim),
                (axes[1],spread,'seed range of Re K','magma',0,max(1e-9,np.max(spread)))):
                im=ax.imshow(data,cmap=cmap_name,vmin=vmin,vmax=vmax)
                for a in range(3):
                    for b in range(3):
                        suffix='*' if ax is axes[0] and not stable[a,b] else ''
                        text_color=('black' if abs(data[a,b])<.45*lim else 'white') if ax is axes[0] else (
                            'black' if data[a,b]>.6*vmax else 'white')
                        ax.text(b,a,f'{data[a,b]:.3g}{suffix}',ha='center',va='center',color=text_color,fontsize=8)
                ax.set(xticks=range(3),yticks=range(3),xticklabels=['low','mid','high'],yticklabels=['low','mid','high'],
                       title=f't={gate["t"]:g}, k={gate["k"]}: {label}'
                       +('\nOne seed: stability untested' if seed_count<2 else ''))
                fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
        save(fig,'accepted_coherent_kernels')

    from fk.runtime import save_json
    save_json(output/'comparisons.json',{'costs':cost_summary,'interpretable_histogram_bins':histogram_masks,
        'interpretable_haar_reference_bins':haar_masks,'coherent_kernel_seed_ranges':kernel_ranges,
        'development_diagnostic':development_diagnostic,'configured_seed_count':seed_count,
        'equal_candidate_counts_not_equal_walltime':True,'canonical_B_is_not_guided_U':True})
    lines=['# FK canonical 分解：构型发现与相干检验',
      '',('**DEVELOPMENT DIAGNOSTIC：当前配置不同于正式方案，只用于开发诊断，不能作为正式三种子实验的结论。**'
          if development_diagnostic else '**正式固定配置实验。**'),
      '',f'运行状态：`{summary.get("status","unknown")}`。配置含{len(config["t_values"])}个k=2物理点及{len(config["k4_t_values"])}个k=4对照点。',
      '',f'跨配置的全部{seed_count}个种子的验收：重构 {acceptance.get("reconstruction_passed_count",0)}/{candidate_count}；'
      f'分布解释 {acceptance.get("distribution_allowed_count",0)}/{candidate_count}；相干解释 {acceptance.get("coherent_allowed_count",0)}/{candidate_count}。',
      '', '**证据分开计量：** 确定性整数矩支持 canonical 重构；半整数质量与分布仍需积分；'
      'witness引导的SOCP成本属于另一种轨道表示，不能替换canonical的B。所有成本上界均非全字典最优性或区间认证。',
      '', '## A. 确定性 canonical 重构', '',
      f'代数与完整张量审计通过：`{exact.get("verified",False)}`。实际k=2数值解完整128维重构：',
      '', '| t/V | 完整残差 | 梯度 | 通过 |', '|---:|---:|---:|---|']
    for r in exact['rows']:
        lines.append(f'| {r["t"]:g} | {num(r.get("full_reconstruction_residual"))} | {num(r.get("gradient_norm"))} | {r.get("verified",r.get("reconstruction_accepted",False))} |')
    lines+=['','确定性k=2的误差与下图的随机积分残差是不同量。raw和orbit-average使用同一witness及同一批样本，配对比较没有更换目标。',
            '', '![重构与系数质量结构诊断](figures/canonical_diagnostics.png)',
            '', f'图中区间为配置中{seed_count}个种子的实际范围，不是置信区间；连线仅连接实际计算点。空心点未通过分布解释门槛。'
            +('当前只有一个种子，不能检验跨种子稳定性。' if seed_count<2 else ''),
            '', '## B1. Canonical 分布与有限阶成本', '',
            '| t/V | k | B中位数 | raw残差范围 | RB残差范围 | 最小ESS | eta均值范围 | 分布解释 | 相干解释 |',
            '|---:|---:|---:|---|---|---:|---|---|---|']
    for gate in acceptance['rows']:
        group=[r for r in canonical if r['id']==gate['id']]
        lines.append(f'| {gate["t"]:g} | {gate["k"]} | {num(median(group,"B"))} | {extent(group,"raw_residual")} | '
                     f'{extent(group,"rb_residual")} | {num(gate["minimum_final_ess"])} | {extent(group,"mean_eta")} | '
                     f'{gate["distribution_interpretation_allowed"]} | {gate["coherent_interpretation_allowed"]} |')
    lines+=['','Gaussian探针校准全部通过：`'+str(acceptance.get('probe_calibration_passed',False))+'`。'
            '校准检验已知均值与采样波动是否一致，不保证0.01精度。有限k的Gaussian目标本来就有B>1，故B不是true extent的直接读数。',
            '', f'最终独立流固定N={config["final_samples"]}，总失败概率预算{failure_probability:g}分配给{event_count}个预定事件。'
            '保守统计半径和超预算状态保存在各validation JSON；经验稳定性、ESS和误差门槛均不自动成为统计认证。',
            '', f'最终非探针重构C通过保守统计误差包络的数量：{summary.get("statistically_certified_count",0)}/{candidate_count*seed_count}。'
            '此计数仅针对重构C，不代表边缘分布、各组向量或相干核同时得到认证。']
    if allowed:
        lines+=['','![通过稳定性门槛的边缘分布](figures/accepted_marginals.png)',
                '', f'每一对图使用相同色尺，分别显示canonical系数质量和同批样本的Haar参考概率。'
                f'灰色格子的局部ESS在至少一个种子中低于{config["bin_ess_min"]:g}；canonical使用加权ESS，Haar参考使用该格子的实际样本数。'
                '两图各自遮罩，灰色不表示零概率。这些是有限阶canonical分布，不能据此声称最优分解的支持转变。'
                'Haar参考直方图与TV距离保存在原始数组/JSON。']
    else:
        lines+=['','没有通过分布解释门槛的结果，不绘制物理解释用的分布图。原始诊断全部保留。']
    lines+=['','## B2. 不输入弧族的构型发现', '',
      f'引导组从与随机组相同的{atom_count}个Haar原子出发，用固定k=2 witness和全部120个Majorana双线性平面做局部重叠优化。'
      f'人工弧族只在独立的物理先验组使用。每组128个公共Fock原子加{atom_count}个新原子；候选数相同，计算时间不同。',
      '', '| t/V | 随机U中位数 | 引导U中位数 | 弧族U中位数 | 引导≤1.05×弧族的种子数 | 引导严格优于随机的种子数 |',
      '|---:|---:|---:|---:|---:|---:|']
    for r in cost_summary:
        lines.append(f'| {r["t"]:g} | {num(r["random_median"])} | {num(r["guided_median"])} | {num(r["physics_arc_median"])} | '
                     f'{sum(x["guided_within_5pct_prior"] for x in r["paired"])} / {len(config["seeds"])} | '
                     f'{sum(x["guided_strictly_below_random"] for x in r["paired"])} / {len(config["seeds"])} |')
    lines+=['','![等候选数成本比较](figures/discovery_costs.png)',
            '', 'U包含完整Fock残差修补及目标态误差估计。局部最大重叠不提供全字典对偶上界；SOCP最优仅限候选轨道。'
            '轨道代表不是有限Gaussian分支。图右侧属于SOCP轨道表示，不能与canonical分布混称同一结果。',
            '', '| 方法 | 候选生成时间合计(s) | SOCP时间合计(s) |', '|---|---:|---:|']
    for method in ('random','guided','physics_arc'):
        r=[r for r in discoveries if r['method']==method]
        generation=sum(x['candidate_generation_seconds'] for x in r)
        if method=='physics_arc' and config['t_values']:
            generation/=len(config['t_values'])
        lines.append(f'| {method} | {generation:.4g} | {sum(x["coefficient_solve_seconds"] for x in r):.4g} |')
    lines+=['','弧族生成在各t之间复用，上表只计一次；guided时间含Haar初始化，未包含共用的canonical witness准备成本。'
            '完整端到端耗时见summary.json，不能从等候选数推断等时间优势。',
            '', '## C. 相干贡献', '']
    if coherent:
        lines+=['![通过验收的相干核](figures/accepted_coherent_kernels.png)',
                '', '核由独立半样本交叉估计。每行左图显示实部的跨种子中位数，右图显示实部最大值减最小值；这些范围不是置信区间。'
                '星号表示跨种子符号不一致、范围包含零，或只有一个种子而无法检验稳定性；不对这些格子推断相长或相消。'
                '只展示完整闭合、CQ/W精度、每组向量RMSE和半样本一致性都通过的点。'
                '通过总体验收不保证每个矩阵元素稳定；实部与虚部的逐元素范围均保存于comparisons.json。'
                '仅有这些点不能定位关联峰的成因或新的临界点。',
                '', '| t/V | k | 最大Re K范围宽度 | 最大Im K范围宽度 | 跨种子实部同号格数 |',
                '|---:|---:|---:|---:|---:|']
        for gate in coherent:
            stats=kernel_ranges[gate['id']]
            lines.append(f'| {gate["t"]:g} | {gate["k"]} | '
                f'{num(np.max(stats["real_range"]))} | {num(np.max(stats["imag_range"]))} | '
                f'{np.sum(stats["real_sign_consistent_across_seeds"])} / 9 |')
    else:
        lines+=['没有结果通过全部相干解释门槛。不能由总重构或边缘分布稳定推断各分组相干贡献准确；本轮不解释Wick峰的来源。']
    lines+=['','| t/V | k | 最大闭合残差 | 最大CQ误差 | 最大W误差 | 最大组RMSE | 最大半样本差异 |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    for gate in acceptance['rows']:
        group=[r for r in canonical if r['id']==gate['id']]
        if not group:continue
        fields=[max(r[f] for r in group) for f in ('coherent_closure_residual','CQ_error','W_error')]
        fields += [max(max(r[f]) for r in group) for f in ('group_rmse','group_half_disagreement')]
        lines.append('| '+f'{gate["t"]:g} | {gate["k"]} | '+' | '.join(num(x) for x in fields)+' |')
    lines+=['','## D. 算法引导成本的必要性检验', '',
            f'| t/V | eta上限 | 受限字典下界 | 引导上界最大值 | 配置的全部{seed_count}个种子均分离 |', '|---:|---:|---:|---:|---|']
    for t in config['t_values']:
        group=[r for r in discoveries if r['t']==t and r['method']=='guided']
        for eta in config['discovery']['eta_thresholds']:
            certs=[next(c for c in r['certificates'] if c['eta0']==eta) for r in group]
            if certs:
                lines.append(f'| {t:g} | {eta:g} | {min(c["restricted_cost_lower"] for c in certs):.6g} | '
                             f'{max(c["cost_upper"] for c in certs):.6g} | '
                             f'{len(certs)==len(config["seeds"]) and all(c["numerical_separation"] for c in certs)} |')
    lines+=['','这些比较使用引导组自己的修补上界，未借用人工弧族成本。正分离说明同等低成本的精确表示无法满足该关联限制；'
            '它不证明canonical分布已近最优，也不是区间算术认证。必要平均eta及高eta系数质量比例详见discovery JSON。',
            '', '## 可复现文件', '',
            '- `config.json`、`manifest.json`、`requirements-lock.txt`：冻结参数、来源与依赖。',
            '- `canonical_k2.*`、`training_*.*`、`frozen_*.npz`：矩证明、完整重构、经验训练及冻结witness。',
            '- `development_*.*`、`validation_*.*`：独立流统计、完整累加器、分布和相干核。',
            '- `discovery_*.*`：原始Gaussian原子、局部旋转路径、相位、SOCP系数和必要性界。',
            '- `acceptance.json`、`comparisons.json`、`summary.json`：所有门槛、配对比较和实际资源。',
            '- `verification.json`：独立保存数组复核；不重做Monte Carlo或系数优化。',
            '', '本轮未增加系统规模或扫描密度，未修改主库或先前实验源文件。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return {'report':str(output/'REPORT.md'),'figures':str(figdir)}
