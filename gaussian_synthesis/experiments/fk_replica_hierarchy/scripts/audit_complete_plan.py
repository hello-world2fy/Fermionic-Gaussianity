#!/usr/bin/env python3
"""Require concrete E0–E5 artifacts and write the final scientific report.

Completion means the declared experiments and explicit qualified conclusions,
not that every hoped-for theorem or algorithmic advantage was established.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4.io import write_json,source_manifest


def read(path):return json.loads(Path(path).read_text())


def verified(directory):
    v=read(Path(directory)/'verification.json');assert v['artifact_verification_passed'],directory
    return v


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scan-state',type=Path,default=Path('results/finalize_scan_01_state.json'))
    p.add_argument('--transfer-state',type=Path,default=Path('results/transfer_pipeline_01_state.json'))
    p.add_argument('--test-log',type=Path,required=True);args=p.parse_args()
    scan=read(args.scan_state);transfer=read(args.transfer_state)
    assert scan['status']==transfer['status']=='complete'
    test_text=args.test_log.read_text();assert '\nOK\n' in test_text and 'FAILED' not in test_text
    hierarchy=read(Path(scan['figure'])/'observations.json')
    assert hierarchy['complete_declared_E1_scan'] and hierarchy['all_reconstruction_and_mass_gates_pass']
    rows=hierarchy['rows'];points={(r['t'],r['k']) for r in rows};assert len(points)==len(rows)==55
    for r in rows:
        assert r['gate_pass'] and r['rho']<1e-3 and r['B_95_halfwidth']/r['B']<.01
        assert r['U_best_strictly_certified'] and r['P_saved']
    e0dirs=['results/e0_gpu_householder_magic4_n131072_01','results/e0_gpu_householder_gaussian4_n131072_01','results/e0_gpu_householder_gaussian8_n524288_01']
    e0=[]
    for directory in e0dirs:
        verified(directory);s=read(Path(directory)/'summary.json')
        assert len(s['comparisons'])==16 and all(r['one_percent_empirical_gate'] and r['all_purity_pass'] for r in s['comparisons'])
        e0.append({'run':directory,'checks_passed':16,'maximum_relative_repeat_halfwidth':max(r['repeat_95_percent_halfwidth'] for r in s['comparisons'])})
    for k in [2,3,4]:verified(f'results/cartan_exact_k{k}_01')
    for directory in [scan['blocks'],scan['windows'],'results/columns_n32_s5_01','results/column_cost_certificates_01',
                      'results/product_magic_canonical_gpu_n131072_01','results/product_magic_asymptotics_01',
                      'results/energy_windows_01','results/fixed_geometry_n8192_01',transfer['predictions']]:verified(directory)
    predictions=Path(transfer['predictions']);ps=read(predictions/'summary.json');coverage=read(predictions/'comparison_coverage.json')
    assert coverage['complete_required_replica_matrix'] and coverage['all_frozen_nonoracle_predictions_precede_ED']
    assert len(ps['rows'])==10650
    methods=set(coverage['all_methods'])
    assert {'random','physics_prior','dual_columns','single_gaussian_HFB','husimi_test_oracle','husimi_localmax_test_oracle','husimi_train_frozen',
            'replica_k2_P','replica_k8_P','replica_k16_P','replica_k32_P'}<=methods
    for r in ps['rows']:
        assert r['actual_gaussian_terms']==r['budget'] and r['actual_gaussian_terms'] in [1,2,4,8,16,32,64]
        assert r['energy_error']>-1e-7 and 0<=r['fidelity']<=1+1e-10
        if 'oracle' not in r['method']:assert not r['uses_test_target']
    costs=read(predictions/'cost_audit.json');assert len(costs['rows'])==len(ps['rows'])
    aggregates=read(predictions/'aggregates.json')
    window=read(Path(scan['windows'])/'summary.json')
    exact_window=[r for r in window['rows'] if r.get('cost_threshold_kind')=='certified_constructive_upper' and abs(r['delta']-.01)<1e-15]
    endpoint=read('results/product_magic_canonical_gpu_n131072_01/summary.json')
    verified('results/product_witness_transfer_01')
    product_transfer=read('results/product_witness_transfer_01/summary.json')
    e128=next(r for r in endpoint['rows'] if r['k']==128)
    dependencies=[args.scan_state,args.transfer_state,Path(scan['figure'])/'observations.json',predictions/'verification.json',predictions/'comparison_coverage.json',predictions/'cost_audit.json',args.test_log]
    dependencies += [Path('results/product_witness_transfer_01/summary.json'),Path('results/product_witness_transfer_01/verification.json')]
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    limitations=[
        'High-order canonical L, numeric radial scales and B integration errors remain empirical, not interval certified; separate conservative exact-block witnesses and exact-Gaussian repaired U are certified.',
        'Nonzero-t global optimal contact manifolds and tight optimality gaps are unresolved. Local Hessians/geometry fits are qualified; no contact transition at t=10 is claimed.',
        'HFB uses32 starts of a local complete-manifold optimizer, not a global certificate; incoherent comparison uses normalized |c|² weights, not optimized mixture weights.',
        'Cold runtime sums recorded source-run walltimes with shared dependencies counted once; not an equal-time hardware benchmark. Interpreter startup and unrecorded setup remain outside timing.',
        'The t0 asymptotic coefficient22 is proved from the normal Hessian, but no explicit finite-k remainder bound is provided.',
        'The optional second physical model and optional extra nonzero-t k128 scans were not required and were not run.'
    ]
    audit={'status':'complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'declared_experiment_plan_executed':True,'all_hoped_for_scientific_claims_proved':False,
        'E0_analytic_checks':e0,'E1_physical_order_points':len(rows),'E1_all_empirical_gates_pass':True,
        'E2_independent_column_runs':45,'E2_t0_exact_global_support_and_necessity':True,
        'E2_conservative_global_dual_and_constructive_upper_certificates':True,
        'E3_fixed_reference_points':[0,9,11,20],'E3_fixed_geometry_maximum_k':128,
        'E3_energy_window_constructions':336,'E3_high_order_window_certificate_rows':len(window['rows']),
        'E3_analytic_endpoint_transfer_window_certificates':len(product_transfer['window_rows']),
        'E4_E5_predictions':len(ps['rows']),'E4_E5_complete_replica_comparison_matrix':True,
        'empirical_reconstruction_threshold':.001,'empirical_relative_B_repeat_threshold':.01,
        'scientific_limitations':limitations,
        'dependency_sha256':{str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in dependencies}}
    write_json(args.output/'audit.json',audit)
    lines=['# FK replica hierarchy：E0–E5实验报告','',
        '方案规定的分级扫描、独立Gaussian构造、几何/能窗实验、完整相干核和冻结字典扰动对照均已执行并复核。完成的是实验与有证据范围的结论；非零t的全局最优接触分类、高阶canonical积分的严格认证仍未解决。本报告主要展示系数l1成本tau_G；Gaussian extent为xi_G=tau_G²。',
        '', f"本轮共55个(t,k)扫描点、48项GPU解析回归、10650个最终公平对照预测。所有正式扫描点均通过完整表示重构残差<10⁻³与B五种子相对重复半宽<1%的经验门槛；这些门槛不等于严格统计覆盖。",'',
        '## 最主要的结果','',
        f"1. **高阶收敛真实但有限阶仍慢。** t=0到k=128时，canonical B={e128['B_mean']:.8g}±{e128['B_95_percent_halfwidth']:.3g}，比精确最优成本2高{100*(e128['B_mean']/2-1):.3f}%。其canonical L点估计为{e128['L_canonical_point_mean']:.8g}，不称为认证下界。独立全局支持证书仍给L=U=2。", 
        '2. **几何解释了端点的收敛系数。** t=0完整Gaussian切空间Hessian在Thouless坐标下精确为0×12、1×32、2×12；44个正法向方向给出B_k=2+22/k+O(k⁻²)。k(B_k−2)的独立采样向22靠近。t=0的所有最优分解必须支撑于S⁶×S⁶；低成本分解还有定量eta约束。',
        '3. **物理必要性可以严格证明，但解析参考更强。** 高阶方向改善了同类q4归一化证书；已知端点w0=2M_AM_B还能跨参数使用，与能量不等式结合后给出更强的能窗排除能力。不能把只优于弱归一化基线的结果说成高阶独有优势。',
        '4. **高阶改善了有限字典，但未超过强基线。** 在epsilon=.2、32个实际Gaussian项下，六个t的平均保真度均随k2→8→16→32上升；k32仍全部低于物理先验与列生成。包括上游采样成本后，本轮不支持优于这些方法的算法优势。','',
        '## E1：成本与分级覆盖','',
        f"![Replica hierarchy]({scan['figure']}/figure1_progress.png)",'',
        '| t | 已测k | 最高k的B ± 经验95%半宽 | 该点canonical L（高阶未认证） | 独立严格U |',
        '|---:|---|---:|---:|---:|']
    for t in sorted({r['t'] for r in rows}):
        group=sorted([r for r in rows if r['t']==t],key=lambda r:r['k']);last=group[-1]
        lines.append(f"| {t:g} | {', '.join(str(r['k']) for r in group)} | {last['B']:.8g} ± {last['B_95_halfwidth']:.3g} | {last['L']:.8g} | {last['U_best']:.9g} |")
    lines += ['', '把解析端点witness用于所有目标，并与精确k4及tau≥1取最强下界，另得到以下完整Gaussian字典的严格区间；它们不是高阶canonical L的认证。',
              '', '| t | 已认证tau下界 | 已认证tau上界 | G=(U−L)/U | 任意B≤U分解的平均eta上界 |',
              '|---:|---:|---:|---:|---:|']
    for r in product_transfer['rows']:
        lines.append(f"| {r['t']:g} | {r['best_combined_physical_cost_lower']:.9g} | {r['U_certified']:.9g} | {r['certified_global_relative_gap_upper']:.6g} | {r['mean_eta_upper_all_cost_le_U']:.6g} |")
    lines += ['', '图中空心L是canonical积分点估计；绿色三角是另行精确q4归一化后的保守高阶可行witness。二者不是同一个归一化，不能混用来声称小的严格缺口。黑线采用精确t0上界2或有IEEE舍入包围的相邻精确Gaussian修补构造成本。',
        '', '数值半径通过符号恒等式u_eff=(Re<u|psi>/q_k(u))^(1/(2k−1))u消元；完整角向重构仍独立核验。Spin(7)群平均对应真实Gaussian酉轨道，不把平均投影态计为Gaussian原子。',
        '', '## E2/E3：必要性与几何','',
        't=0尖锐支持界为|<2M_AM_B|g>|≤1−eta/2。因此U≤2.02的所有精确分解中，eta≥0.2的系数质量最多约9.901%；精确最优U=2迫使eta=0和两个局部magic接触条件。',
        '', '以下取Delta=.01，U为已认证的独立构造上界；数字是能窗内系数质量比例的严格上界。1表示此方向没有排除信息，filled是另行给出的参考witness。',
        '', '| t | U | k2 | k4 | k8 | k16 | k32 | k64 | filled | 解析端点w0 |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in [9,9.1,11,20]:
        rs={r['source_direction_order']:r for r in exact_window if r['t']==t}
        values=[f"{rs[k]['fraction_upper']:.6g}" if k in rs else '—' for k in [2,4,8,16,32,64,0]]
        analytic=next(r for r in product_transfer['window_rows'] if r['t']==t and abs(r['delta']-.01)<1e-15)
        values.append(f"{analytic['fraction_upper']:.6g}")
        lines.append(f"| {t:g} | {next(iter(rs.values()))['declared_cost_threshold']:.9g} | "+' | '.join(values)+' |')
    lines += ['', '表格最后一列来自解析端点支持和eta≥(1−X_Delta)²，明显强于本轮高阶q4参考；其受限支持由显式质量/hopping弧达到。以下图仅展示相同q4归一化方向之间的比较。证明与完整参考表见[跨参数必要性](PRODUCT_WITNESS_TRANSFER.md)及[数值证书](results/product_witness_transfer_01/REPORT.md)。',
        '',f"![能窗必要性]({scan['windows']}/window_necessity.png)",'',
        '固定witness几何在t=0、9、11、20测到k128。t0拟合有效维数约11.91与严格维数12相符；t9/11存在约10⁻⁶软曲率，尚不足以分类真实极限接触流形；t20有限阶有效维数约6.82，未替代全局最大值认证。能窗构造的336组数据独立检验了t9的Delta⁻¹、t>9的Delta⁻²障碍及误差依赖。',
        '',f"![Canonical组成]({scan['composition']}/canonical_composition.png)",'',
        '## E4/E5：冻结字典、相干核与扰动预测','',
        '使用t=9、10、10.9225、11、12、20；epsilon=0、.02、.05、.1、.2；实际Gaussian项数1、2、4、8、16、32、64；每格五个重复。k2/8/16/32字典及普通基线全部在扰动ED前冻结，系数只由新Hamiltonian的H/S广义本征问题求得。直接目标Husimi分布与局部极大值测试oracle明确使用ED目标。',
        '', '以下为epsilon=.2、32个实际Gaussian项的平均保真度（HFB仅1项）：','',
        '| t | k2 | k8 | k16 | k32 | 物理先验 | 列生成 | 随机 | Husimi oracle | HFB(1项) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    comparison_methods=['replica_k2_P','replica_k8_P','replica_k16_P','replica_k32_P','physics_prior','dual_columns','random','husimi_test_oracle','single_gaussian_HFB']
    for t in [9,10,10.9225,11,12,20]:
        values=[]
        for m in comparison_methods:
            r=next(r for r in aggregates if r['t']==t and r['epsilon']==.2 and r['method']==m and r['actual_terms']==(1 if m=='single_gaussian_HFB' else 32))
            values.append(f"{r['fidelity_mean']:.6g}")
        lines.append(f"| {t:g} | "+' | '.join(values)+' |')
    example=next(r for r in aggregates if r['t']==11 and r['epsilon']==.2 and r['method']=='replica_k32_P' and r['actual_terms']==32)
    peak=next(r for r in aggregates if r['t']==10.9225 and r['epsilon']==0 and r['method']=='replica_k32_P' and r['actual_terms']==32)
    peak_target=next(r for r in ps['targets'] if r['t']==10.9225 and r['epsilon']==0)
    lines += ['',f"例如t11的k32平均保真度为{example['fidelity_mean']:.6f}，五重复经验95%半宽为{example['fidelity_95_halfwidth']:.3g}。这表明候选组成随阶数改进；表中的物理先验和列生成仍更准确，不能据此宣称高阶采样是更有效的预测算法。",
        '',f"![扰动保真度]({predictions}/figures/transfer_fidelity.png)",'',
        f"![完整相干核与非相干对照]({predictions}/figures/coherent_wick_comparison.png)",'',
        f"在Wick峰附近t=10.9225、epsilon=0、32项时，目标偏离为{peak_target['metrics']['wick_deviation']:.6g}，k32相干态给{peak['coherent_wick_deviation_mean']:.6g}，同原子非相干混合仅给{peak['incoherent_wick_deviation_mean']:.6g}。这量化了该有限字典中相干干涉的贡献。高阶观测量误差不随k普遍单调：例如t20的k16 Wick误差高于k8；更高保真度本身不保证每个观测量都更准。",'',
        'Hamiltonian、质量关联和三个四点算符均保存完整Fock核，并独立用256维算符、SVD子空间能量和显式Wick收缩复核。非相干对照采用同一原子的|c|²归一化权重，未优化混合权重，不能宣称是所有Gaussian混合的最优限制。HFB是32起点局部优化；保留秩与真实项数分别报告。',
        '', f"全部预算和扰动的能量、关联误差及重复区间见[{predictions.name}/REPORT.md]({predictions}/REPORT.md)。成本采用[cost_audited_aggregates.csv]({predictions}/cost_audited_aggregates.csv)：递归计入精确矩、前阶初始化、训练和验证采样；共享来源一次计入，物理先验按相同候选的独立构造计时。此为记录到的源运行墙钟总和，不是等时硬件竞赛，也未计入未记录的启动开销。",
        '', '## 验证、失败记录与复现','',
        f"完成审计及依赖哈希：[audit.json]({args.output}/audit.json)。测试日志：[{args.test_log.name}]({args.test_log})。每个实验目录均保存运行配置、源码快照、SHA256、原始数组与独立verification.json。", 
        '', '旧k4随机积分重构失败由精确Cartan后端修复；旧k8数值半径失败由另行记录的径向消元与新验证处理，没有重标旧结果。Gaussian8 GPU高阶重复误差曾超过1%，增加到524288粒子后通过。t20k64原串行01在两个训练种子后为四卡并行02中止，partial文件和interruption.json保留；134个小规模训练/验证数组与串行逐位一致。',
        '', '使用8×H20 96GB及允许的174个逻辑CPU；用户授权的旧占卡任务已通过STOP正常退出。未提交或推送Git变更。第二物理模型和额外非零t的k128是条件扩展，本轮未执行。',
        '', '严格证明与实现口径：[Cartan精确矩](CARTAN_THEORY.md)、[高阶分块界](CARTAN_BLOCK_BOUNDS.md)、[径向消元](RADIAL_ELIMINATION.md)、[t0必要性](PRODUCT_MAGIC_NECESSITY.md)、[跨参数必要性](PRODUCT_WITNESS_TRANSFER.md)、[t0渐近系数](PRODUCT_MAGIC_ASYMPTOTICS.md)、[Gaussian舍入修补](GAUSSIAN_ROUNDING_CERTIFICATES.md)、[能窗定理](ENERGY_WINDOW_THEORY.md)。']
    timing=['','## 记录到的成本分解','',
        f"完整最终对照的唯一上游依赖、冻结和预测运行计时之和为{costs['all_methods_recorded_source_freeze_prediction_seconds']:.3f}秒（{costs['all_methods_recorded_source_freeze_prediction_seconds']/3600:.3f}小时）。并行源运行的时间在这里相加，不是用户实际等待时间；不包含开发失败、独立复核和未记录初始化。",'',
        '以下取t11、epsilon=.2、32项（HFB1项）的五重复平均。冷启动上游采用完整可复用来源任务计时，未将其他点或预算均摊掉；其范围与每次预测耗时不同。',
        '', '| 方法 | 上游来源秒 | 候选生成秒 | 选择秒 | 单次预测秒 | 冷启动记录合计秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for method in comparison_methods:
        selected=[r for r in costs['rows'] if r['t']==11 and r['epsilon']==.2 and r['method']==method and r['actual_gaussian_terms']==(1 if method=='single_gaussian_HFB' else 32)]
        assert len(selected)==5
        timing.append('| '+method+' | '+' | '.join(f"{np.mean([r[field] for r in selected]):.6g}" for field in ['audited_recorded_source_seconds','candidate_generation_seconds','selection_seconds','prediction_seconds','audited_recorded_cold_total'])+' |')
    lines+=timing
    Path('REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Audited55 E1 points,48 E0 checks,10650 E4/E5 predictions; report written.')


if __name__=='__main__':main()
