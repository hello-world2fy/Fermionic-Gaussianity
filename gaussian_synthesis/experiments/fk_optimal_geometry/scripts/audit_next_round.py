#!/usr/bin/env python3
"""Requirement-level completion audit; open ideal results stay explicitly open."""
from pathlib import Path
from fractions import Fraction
import datetime
import hashlib
import json
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from fk5.io import write_json
from fk5.paths import ROOT,PREVIOUS


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert (ROOT/'EXPERIMENT_SPEC.txt').read_bytes()==Path('/home/tiger/.codex/attachments/791792a9-e809-48a1-bcb1-72b46400c576/pasted-text-1.txt').read_bytes()
    assert not subprocess.check_output(['git','status','--porcelain','--',str(PREVIOUS)],cwd=ROOT).strip()
    evidence={};requirements=[]
    def verified(run):
        path=ROOT/'results'/run/'verification.json';value=read(path);assert value['artifact_verification_passed']
        evidence[str(path.relative_to(ROOT))]=sha(path)
        if 'summary_sha256' in value:assert value['summary_sha256']==sha(path.parent/'summary.json')
        return value
    initial=read(ROOT/'results/initial_certificates_02/summary.json');vi=verified('initial_certificates_02')
    assert {r['t'] for r in initial['A1_rows']}=={5,9,10,10.9225,11,12,20}
    c=read(ROOT/'results/two_orbit_02/summary.json');vc=verified('two_orbit_02')
    assert vc['actual_gaussian_terms']==16 and vc['distinct_Gaussian_rays']==16 and not vc['no_Fock_repair'] is False
    assert vc['certified_relative_gap_upper_with_A1']<.01 and not c['primal_dual_matching_proved']
    assert c['starts']==128 and len(c['contacts'])==2
    # Independently recompute the deliberately conservative matching-candidate
    # global support envelope; a loose result is retained, not called matching.
    mp.dps=110;r=read(ROOT/'results/matching_candidate_support_02/summary.json');z=mp.matrix([mp.mpf(v) for v in r['frozen_real_z']])
    V=mp.matrix([[mp.mpf(1)/4,mp.sqrt(14)/4,mp.mpf(1)/4],[-mp.sqrt(2)/2,0,mp.sqrt(2)/2],[mp.sqrt(7)/4,-mp.sqrt(2)/4,mp.sqrt(7)/4]])
    proof=read(PREVIOUS/'results/cartan_exact_k4_01/proof.json');beta=V.T*z;x=[beta[0],beta[1]/mp.sqrt(14),beta[2]]
    monomial=mp.matrix([mp.fprod(v**p for v,p in zip(x,powers)) for powers in proof['powers']])
    gram=mp.matrix([[mp.mpf(v)/mp.mpf(proof['projector_denominator']) for v in line] for line in proof['gram_numerator']]);q4=(monomial.T*gram*monomial)[0]
    assert mp.mpf(r['exact_q4_interval'][0])<=q4<=mp.mpf(r['exact_q4_interval'][1])
    assert mp.mpf(r['q4_support_upper'])>=q4**(mp.mpf(1)/8)
    a,b=[mp.mpf(v) for v in r['joint_decomposition_coefficients']]
    h=(4*a+b/2 if b<=4*a else a+b+2*a*a/(b-2*a))+mp.sqrt((z[0]-a-2*b)**2+(z[1]+mp.sqrt(8)*a)**2+(z[2]-mp.sqrt(7)*a)**2)
    assert mp.mpf(r['joint_plus_Hilbert_remainder_support_upper'])>=h
    assert r['global_support_upper']==min(r['q4_support_upper'],r['joint_plus_Hilbert_remainder_support_upper'])>1
    write_json(ROOT/'results/matching_candidate_support_02/verification.json',dict(artifact_verification_passed=True,
        independent_decimal_digits=110,exact_q4_and_joint_remainder_bounds_checked=True,matching_support_one_proved=False,
        summary_sha256=sha(ROOT/'results/matching_candidate_support_02/summary.json'),verification_script_sha256=sha(__file__)))
    verified('matching_candidate_support_02')
    requirements.extend([
        ('A1','complete','Seven exact-spectrum/target certificates and independently verified t20 relative gap <1%.'),
        ('A2','bounded_study_complete_ideal_open','Actual 16-term certificate, 128-start discovery and conservative global support attempt complete; matching support 1 not proved.')])
    v=verified('contact_comparison_01');assert v['tighter_eta_and_concentration_bounds_checked'] and not v['global_quotient_distance_stability_constant_certified']
    contact=read(ROOT/'results/contact_comparison_01/summary.json');assert contact['contact_only_exact_reconstruction_impossible']
    for path in ['JOINT_CONTACT_THEORY.md','JOINT_WITNESS_THEORY.md','CONTACT_ASYMPTOTICS.md']:assert (ROOT/path).stat().st_size>1000
    requirements.extend([('B1','complete','Global joint contact equality proof, finite atoms, active columns, quotient-invariant diagnostics and contact-only obstruction.'),
        ('B2','first_stage_complete_strengthening_open','All-decomposition eta mean/tail bounds and saturation slack verified; global orbit-distance c and radius remain explicitly unproved.')])
    v=verified('fixed_gpu_n65536_s5_01');assert v['sample_checkpoints_checked']==225 and v['independent_runs']==25
    verified('soft_curvatures_01');v=verified('two_contact_toy_01');assert v['sample_checkpoints_checked']==70
    v=verified('high_order_direction_certificates_01');assert len(v['directions_checked'])==8 and v['coefficient_measure_feature_arrays_checked']==40
    assert len(v['strictly_excluded_from_beating_joint'])==6
    requirements.extend([('C1','complete','Fixed-order grid, independent seeds, 50/90-digit Hessians and exact soft-curvature identities.'),
        ('C2','complete_with_conditional_FK_limit','Known-optimum toy weights and 1/k drift verified; moving FK sequence tested and found outside proposed asymptotic balance.'),
        ('C3','bounded_study_complete_no_improvement','Eight requested directions, two certification routes, deterministic moment envelopes and six rigorous stopping exclusions; no accurate high-order point certificate claimed.')])
    v=verified('window_sharp_bounds_02');assert v['bounds_checked']==48
    v=verified('window_variational_01');assert v['normalized_trial_states_checked']==96 and sum(r['count'] for r in v['dictionaries'])==1164
    assert {(r['t'],r['delta'],r['budget_multiplier']) for r in v['comparisons']}=={(t,d,b) for t in [9,11,20] for d in [.001,.01,.1,1.] for b in ['1','1.1','1.5','2']}
    v=verified('energy_exponents_01');assert v['FK_constructions_checked']==36 and len(v['root_of_unity_full_Fock_constructions'])==4
    requirements.extend([('D1_D2','complete','All 48 strict bounds, full-spectrum gap, 96 normalized budget trials and sharper global bounds, with solver warnings and local-search limits retained.'),
        ('D3','complete','Uniform all-minima r/q theorem, functional-norm error factor, matching finite FK constructions, and a sharp-leading-constant toy.')])
    v=verified('structure_ablation_01');assert v['solver_configurations_checked']==720 and v['normalized_states_checked']==2880
    assert v['full_scope']==[[t,R] for t in [9,10.9225,11,20] for R in [2,4,8,16,32,64]]
    v=verified('coherent_orbits_01');assert sum(r['kernels_checked'] for r in v['checks'])==7760
    requirements.extend([('E1','complete','Four couplings x six actual distinct-ray budgets; contacts, deletions, low-energy, prior, columns and k8/16/32; costs/F/E/mass/Wick retained.'),
        ('E2','complete','Full deterministic two/four-point kernels, within/between-orbit parts, normalized positive-density comparisons and arbitrary-term orbit/window exclusions.')])
    run=ROOT/'results/new_perturbations_01';reg=read(run/'preregistration.json');pred=read(run/'prediction_summary.json');barrier=read(run/'ED_ACCESS_BARRIER.json')
    assert pred['status']=='complete' and len(pred['rows'])==160
    assert reg['amplitudes']==[.02,.05,.1,.2] and reg['pattern_replicates']==5
    assert {c['kind'] for c in reg['cases']}=={'hopping','interaction'}
    assert reg['frozen_utc']<pred['complete_utc']<barrier['first_ED_access_after_utc']
    assert barrier['preregistration_sha256']==sha(run/'preregistration.json') and barrier['prediction_summary_sha256']==sha(run/'prediction_summary.json')
    assert reg['frozen_input_sha256']==sha(run/'frozen_inputs.npz')
    frozen=np.load(run/'frozen_inputs.npz')
    for row in pred['rows']:
        file=run/'predictions'/row['key']/'predictions.npz';assert sha(file)==barrier['prediction_hashes'][row['key']]
        data=np.load(file)
        for result in row['rows']:
            if 'frozen_dictionary_key' in result:
                R=result['requested_maximum_terms'];key=f"{result['method']}_R{R}_atoms"
                np.testing.assert_array_equal(data[key],frozen[result['frozen_dictionary_key']][:R])
    v=verified('new_perturbations_01/evaluation');assert v['predictions_checked']==2400 and v['new_Hamiltonians_checked']==160
    requirements.append(('F1','complete_no_end_to_end_advantage_claim','Both genuinely new perturbations, all frozen parameters, target-free predictions, full ED barrier, HFB and strong adaptive energy-column controls, actual costs/time ledgers.'))
    selection=read(ROOT/'results/extension_candidates_01/selection.json')
    assert selection['selected_model']=='number_conserving_pairing' and selection['selection_before_new_ED']
    assert all(r['full_dimension']==4096 for r in selection['three_cluster_candidate'])
    v=verified('pairing_extension_02');assert v['physical_targets_checked']==3 and v['finite_window_constructions_checked']==12
    # Check the actual 13-term error-q8 truncations, in addition to their
    # independent exact Fourier proof in the extension verifier.
    extension=read(ROOT/'results/pairing_extension_02/summary.json')
    for r in extension['rows']:
        data=np.load(ROOT/'results/pairing_extension_02'/f"g{r['g']:g}_delta{r['delta']:g}.npz")
        error=float(np.linalg.norm(data['coefficients'][:13]@data['atoms'][:13]-data['target']))
        lo,hi=r['drop_eighth_component_Hilbert_error_interval'];assert lo-1e-10<=error<=hi+1e-10
    requirements.append(('F2','complete','Both sparse models prepared; pairing selected before ED; global Gaussian coercivity, full spectral separation and finite exact/approximate exponent-matching constructions verified.'))
    # The source bytes actually used by producers remain recoverable even
    # though the working implementation has evolved during the study.
    snapshots=0
    runs=['initial_certificates_02','two_orbit_02','contact_comparison_01','fixed_gpu_n65536_s5_01',
          'precision_contacts_01','soft_curvatures_01','two_contact_toy_01','high_order_direction_certificates_01',
          'window_variational_01','window_sharp_bounds_02','energy_exponents_01','structure_ablation_01',
          'coherent_orbits_01','new_perturbations_01','extension_candidates_01','pairing_extension_02','matching_candidate_support_02']
    for name in runs:
        folder=ROOT/'results'/name;manifest=read(folder/'manifest.json')
        for relative,digest in manifest['source_sha256'].items():
            assert sha(folder/'source_snapshot'/relative)==digest,(name,relative)
            snapshots+=1
    requirements.append(('DELIVERY','complete','Original plan retained, predecessor read-only, result/verification artifacts and source snapshots audited; failures, seeds and raw arrays kept locally.'))
    audit=dict(execution_scope_complete=True,all_ideal_research_results_achieved=False,
        original_spec_sha256=sha(ROOT/'EXPERIMENT_SPEC.txt'),requirement_checks=[dict(id=i,status=s,evidence_scope=e) for i,s,e in requirements],
        independently_verified_artifact_hashes=evidence,source_snapshot_files_checked=snapshots,
        previous_package_unmodified=True,completion_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        open_ideal_results=['Nonzero-FK exact primal-dual matching','Explicit global quotient-distance stability c and radius',
                           'High-order FK direction with certified improvement over joint witness','End-to-end computational advantage over adaptive energy columns'],
        audit_script_sha256=sha(__file__))
    write_json(ROOT/'results/final_requirement_audit.json',audit)
    lines=['# 原方案逐项验收','',
        '本验收以原方案的执行要求为准。A2 明确允许有限预算内未匹配时保留 <1% 认证；B2 区分第一轮确定性标量必要性结果与希望进一步证明的全局轨道距离常数；C3 要求无改善时停止无条件加阶。开放的理想成果没有被转换成已证明结论。','',
        '| 项目 | 执行状态 | 已检查的证据范围 |','|---|---|---|']
    for item in audit['requirement_checks']:lines.append(f"| {item['id']} | {item['status']} | {item['evidence_scope']} |")
    lines+=['',f'生产运行的 {snapshots} 个源码快照文件逐一匹配记录哈希；此前实验包未修改。机器可读证据及哈希见 [final_requirement_audit.json](results/final_requirement_audit.json)。','',
        '以下理想结果未实现，后续研究可继续，但本轮不无限追加计算：','',
        '- FK 精确最优成本的 primal–dual matching；当前保留严格 0.628301% 区间。',
        '- 全局轨道商距离稳定常数及显式有效邻域；当前严格结论是 η 均值/集中和完整指定轨道的表示障碍。',
        '- 超过联合解析 witness 的高阶 FK 认证；当前指定方向没有获得改善，六个方向被严格排除。',
        '- 优于强自适应列生成的端到端计算优势；扰动对照的负结果与完整成本限制已报告。','',
        '第二模型推广给出了不同 Hamiltonian 上的全局必要性和匹配构造，满足 M4 的理论推广路线。三团簇模型已准备，按预先选择结果不开展缺少相应理论问题的大范围扫描。','']
    (ROOT/'COMPLETION_AUDIT.md').write_text('\n'.join(lines))
    print('Requirement audit passed:',len(requirements),'items;',snapshots,'source snapshot files; ideal open results remain explicit.',flush=True)


if __name__=='__main__':main()
