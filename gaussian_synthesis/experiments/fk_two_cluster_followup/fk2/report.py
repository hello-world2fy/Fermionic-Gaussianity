"""Generate the follow-up report directly from saved numerical results."""
import json
import os
from pathlib import Path
import numpy as np


def _load(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def make_report(output):
    output=Path(output)
    os.environ['MPLCONFIGDIR']=str(output/'.mplconfig')
    os.environ['XDG_CACHE_HOME']=str(output/'.cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output/'figures'
    figures.mkdir(exist_ok=True)
    config=_load(output/'config.json')
    summary=_load(output/'summary.json',{})
    exact=_load(output/'exact_k2.json',{})
    finite=_load(output/'finite_orbits.json',{})
    ablation=_load(output/'ablation.json',{})
    documents=[_load(p) for p in sorted(output.glob('rb_*.json'))]
    rows=[r for doc in documents for r in doc.get('rows',[])]
    max_n=max(config['sample_sizes'])
    final=[r for r in rows if r['samples']==max_n]
    source_final=[r for r in final if r['sourcekind']=='frozen_pilot']
    primary=next((r for r in finite.get('rows',[]) if r['method']=='specific_orbit_simplex_2_design'),None)
    lines=['# FK 双团簇：第二轮实验报告','',
           '本轮研究对称性平均的积分方差、确定性 k=2 矩参考，以及 t/V=20 的显式有限 Gaussian 展开。首轮源码与数据作为冻结输入，原验收结果保持不变。','',
           f"运行状态：`{summary.get('status','unknown')}`。所有数值证据仍使用浮点计算，未作区间算术认证。",'',
           '## 主要结果','']
    if primary:
        lines.extend([f"- 显式构造 **{primary['finite_gaussian_count']} 个 normalized Gaussian ket**，原始完整128维残差 **{primary['raw_full_residual']:.6g}**。",
                      f"- 加上全部 Fock 残差与目标误差估计，成本上界 **{primary['cost_upper']:.12g}**，受限 eta≤0.25 字典下界 **{primary['restricted_cost_lower']:.12g}**；数值分离：**{primary['numerical_separation']}**。",
                      f"- Gaussian 纯性最大误差 {primary['max_gaussian_purity_error']:.4g}；ket 与 covariance lift 最大误差 {primary['max_covariance_lift_error']:.4g}。"])
    if exact:
        lines.append(f"- full 字典 k=2 确定性矩参考通过：**{exact.get('verified',False)}**。它独立计算目标函数与重构，不需要 Monte Carlo 拟合矩阵。")
    lines.extend([f"- 冻结首轮 witness、N={max_n} 的 {len(source_final)} 条记录：普通重构通过 {sum(r['raw_within_tolerance'] for r in source_final)} 条，轨道平均重构通过 {sum(r['orbit_within_tolerance'] for r in source_final)} 条。门槛为 {config['reconstruction_tolerance']}。",'',
                  '**确定性重构、有限展开成本、Monte Carlo 精度和统计置信保证分别报告。经验残差通过不等于联合95%统计认证；本轮没有颁发 Monte Carlo 统计证书。**','',
                  '## 对称性平均与原始重构','',
                  '两种估计使用同一批新样本，首轮 witness 不再调优。轨道平均 Pi g 有单位成本的 Gaussian 群轨道表示，未将 Pi g 归一化或冒充单个 Gaussian 态。补空间噪声的消除来自该表示的解析积分。','',
                  '| 字典 | k | 记录数 | 原始残差中位 | 轨道残差中位 | 估计方差比 raw/orbit 中位 | 原始通过 | 轨道通过 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for name in config['dictionaries']:
        for k in config['orders']:
            a=[r for r in source_final if r['dictionary']==name and r['k']==k]
            if a:
                lines.append(f"| {name} | {k} | {len(a)} | {np.median([r['raw_full_residual'] for r in a]):.6g} | {np.median([r['orbit_full_residual'] for r in a]):.6g} | {np.median([r['paired_variance_ratio'] for r in a]):.5g} | {sum(r['raw_within_tolerance'] for r in a)} | {sum(r['orbit_within_tolerance'] for r in a)} |")
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True,constrained_layout=True)
    for ax,(name,k) in zip(axes.flat,[(n,k) for n in config['dictionaries'] for k in config['orders']]):
        for field,label,color in [('raw_full_residual','Raw Haar','#b84730'),('orbit_full_residual','Orbit averaged','#2367ad')]:
            ns=[];med=[];low=[];high=[]
            for n in config['sample_sizes']:
                vals=[r[field] for r in rows if r['dictionary']==name and r['k']==k and r['samples']==n and r['sourcekind']=='frozen_pilot']
                if vals:
                    ns.append(n);med.append(np.median(vals));low.append(min(vals));high.append(max(vals))
            if ns:
                ax.plot(ns,med,'o-',label=label,color=color)
                ax.fill_between(ns,low,high,color=color,alpha=.12)
        ax.axhline(config['reconstruction_tolerance'],color='black',ls='--',lw=1,label='0.01 threshold')
        ax.set(xscale='log',yscale='log',title=f'{name}, k={k}',xlabel='Fresh validation samples',ylabel='Full ket residual')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Frozen witnesses: median and range across t and seeds (not confidence intervals)')
    for suffix in ('png','pdf'):
        fig.savefig(figures/f'paired_reconstruction.{suffix}',dpi=170)
    plt.close(fig)
    lines.extend(['','![配对重构比较](figures/paired_reconstruction.png)','',
                  '横轴是本轮新增的独立验证样本量；训练量固定为65536。曲线为跨 t 点与种子的中位数，阴影为范围，均不是置信区间。方差比是单样本向量方差迹的估计比；两种方法共享采样，不能由此直接宣称同等倍数的端到端加速。','',
                  '## k=2 确定性参考与训练误差','',
                  'full 字典的对称二次张量空间可用 Gaussian Cartan projector 计算。限制到三维 witness 后得到6×6矩矩阵；本轮另外在完整二次张量空间收缩重构，避免仅通过投影宣告闭合。半整数幂的 B₂ 质量积分仍需独立估计，不能由 q₂ 精确可算推出最小成本已知。','',
                  f'| t/V | 来源 | N={max_n}记录数 | 目标残差中位(raw) | 目标残差中位(orbit) | 确定性重构偏差中位 |',
                  '|---:|---|---:|---:|---:|---:|'])
    for t in config['t_values']:
        for kind in ('frozen_pilot','deterministic_k2'):
            a=[r for r in final if r['dictionary']=='full' and r['k']==2 and r['t']==t and r['sourcekind']==kind]
            if a:
                bias=[r['exact_bias_l2'] for r in a if r.get('exact_bias_l2') is not None]
                b=f'{np.median(bias):.6g}' if bias else '未保存'
                lines.append(f"| {t} | {kind} | {len(a)} | {np.median([r['raw_full_residual'] for r in a]):.6g} | {np.median([r['orbit_full_residual'] for r in a]):.6g} | {b} |")
    probes=[r for r in final if r['sourcekind']=='hopping_gaussian_probe']
    lines.extend(['','解析 hopping Gaussian probe 单独标记为 `hopping_gaussian_probe`，不与物理目标通过数混合；product 字典不使用不适用的 full Gaussian probe 公式。所有 probe 标量参考及误差保存在 rb_full_*.json。',
                  f"最终样本量下 probe 诊断通过 {sum(r.get('probe_diagnostic_calibration_passed',False) for r in probes)}/{len(probes)}；标准为 q/B 各5个解析标准误、raw C为6个解析RMS，均非联合置信保证。",'',
                  '确定性求解的逐点结果（完整张量收缩，不依赖验证样本）：','',
                  '| t/V | Newton步数 | 梯度范数 | 完整128维重构残差 | 完整张量与三维计算差 |',
                  '|---:|---:|---:|---:|---:|'])
    for r in exact.get('rows',[]):
        lines.append(f"| {r['t']:g} | {r['iterations']} | {r['gradient_norm']:.6g} | {r['full_reconstruction_residual']:.6g} | {r['full_tensor_singlet_closure_residual']:.6g} |")
    lines.extend(['',
                  '## 显式有限 Gaussian 展开','',
                  '先合并首轮同相位射线的投影轨道，再在三维空间重拟合系数。针对选中的三个轨道，使用七维质量向量空间中单位球面 S^6 上的8顶点正则单纯形；每个轨道都在完整128维空间验证有限平均等于连续投影。所有分支均保存原始 ket 和复系数，保留共同旋转的相对相位。',''])
    if primary:
        lines.extend([f"主结果使用 {primary['finite_gaussian_count']} 个 Gaussian 分支。完整数值 Fock 修补包含 {primary['fock_repair_count_nonzero']} 个非零系数，其中超过1e-12的有 {primary['fock_repair_count_above_1e_12']} 个；成本包含所有修补系数，不能仅按阈值删去。",
                      f"每轨道有限平均的最大误差为 {max(finite['simplex_orbit_average_errors']):.6g}。这是指定轨道的验证，不声称该设计对任意 Gaussian 原子都适用。",'',
                      '| 观测量 | 原始有限展开归一化后的绝对误差 |','|---|---:|'])
        for name,error in primary['raw_observable_absolute_errors'].items():
            lines.append(f'| {name} | {error:.6g} |')
    lines.extend(['','随机对照沿已选轨道的质量方向球面抽样，并在显式 Gaussian 列与 Fock 修补上统一做 SOCP；这不是全 Gaussian Haar 候选，也没有全局最优性保证。R始终表示总原始 Gaussian 分支数，修补计数另报。','',
                  '| 方法 | R | 重复数 | 修补成本范围 | 原始残差范围 | 数值分离数 |',
                  '|---|---:|---:|---|---|---:|'])
    groups=sorted({(r['method'],r['R']) for r in finite.get('rows',[])})
    for method,R in groups:
        a=[r for r in finite['rows'] if r['method']==method and r['R']==R]
        costs=[r['cost_upper'] for r in a]; errs=[r['raw_full_residual'] for r in a]
        lines.append(f"| {method} | {R} | {len(a)} | {min(costs):.7g}–{max(costs):.7g} | {min(errs):.4g}–{max(errs):.4g} | {sum(r['numerical_separation'] for r in a)} |")
    if finite.get('rows'):
        fig,ax=plt.subplots(figsize=(8,4.6),constrained_layout=True)
        random=[r for r in finite['rows'] if r['method']=='random_spherical_orbit_finite_SOCP']
        sizes=sorted({r['R'] for r in random})
        if sizes:
            med=[np.median([r['cost_upper'] for r in random if r['R']==R]) for R in sizes]
            lo=[min(r['cost_upper'] for r in random if r['R']==R) for R in sizes]
            hi=[max(r['cost_upper'] for r in random if r['R']==R) for R in sizes]
            ax.plot(sizes,med,'o-',label='Random orbit: median',color='#2367ad')
            ax.fill_between(sizes,lo,hi,color='#2367ad',alpha=.15,label='Seed range (not CI)')
        partial=[r for r in finite['rows'] if r['method']=='partial_simplex_finite_SOCP']
        if partial:
            ax.scatter([r['R'] for r in partial],[r['cost_upper'] for r in partial],marker='s',color='#8a6f37',label='Partial simplex + refit')
        if primary:
            ax.scatter([primary['R']],[primary['cost_upper']],s=160,marker='*',color='#b84730',label=f"Constructive {primary['R']}-state design",zorder=5)
            ax.axhline(primary['restricted_cost_lower'],color='black',ls='--',label='Restricted eta<=0.25 lower bound')
        ax.set(xlabel='Total raw Gaussian branches R',ylabel='Cost including Fock residual repair',title='t/V=20: explicit finite Gaussian sums',xscale='log')
        ax.legend(fontsize=8,loc='upper right');ax.grid(alpha=.2)
        for suffix in ('png','pdf'):
            fig.savefig(figures/f'finite_costs.{suffix}',dpi=170)
        plt.close(fig)
        lines.extend(['','![显式有限展开成本](figures/finite_costs.png)'])
    lines.extend(['','## 结构字典消融','',
                  '以下仅比较首轮保存候选的不同子集，均为轨道投影字典。它们用于判断随机 Haar 候选是否改善当前有限问题，不证明结构弧在全字典中全局最优。','',
                  '| t/V | Fock | Fock+结构弧 | Fock+Haar | 全部首轮候选 |',
                  '|---:|---:|---:|---:|---:|'])
    for t in config['t_values']:
        values={r['dictionary']:r['cost_upper'] for r in ablation.get('rows',[]) if r['t']==t}
        cells=[f"{values[n]:.10g}" if n in values else '未完成' for n in ('fock','fock_arc','fock_haar','full_saved')]
        lines.append('| '+str(t)+' | '+' | '.join(cells)+' |')
    lines.extend(['','## 数据、资源与限制','',
                  '- `exact_k2.json/npz`：确定性矩与 witness；`rb_*.json/npz`：配对估计、每个N的完整向量、统计量和源 witness 哈希。',
                  '- `finite_orbits.json/npz`：实际 Gaussian ket、复系数、全部 Fock 修补、covariance与相干观测量。`ablation.json/npz`：候选索引与系数。',
                  '- `manifest.json`：本轮源码、首轮后端、首轮输入文件哈希及依赖版本。`summary.json`：最终阶段状态、总耗时、峰值内存。',
                  '- 后处理只读取保存结果；独立审计在 `verification.json`，单元测试在 `unit_tests.log`。',
                  '- 成本分离是数值可行上界与解析受限下界的浮点评估；没有严格区间证书，也没有完整 Gaussian 字典最优性证明。',
                  '- k=4仍使用首轮经验 witness，若轨道平均后仍未达精度门槛，保留失败结果；本轮不升阶、不增加样本上限、不解释未通过的高阶物理分布。',''])
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return {'report':str(output/'REPORT.md'),'figure_count':2}
