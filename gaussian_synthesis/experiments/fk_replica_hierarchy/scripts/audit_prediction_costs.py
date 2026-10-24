#!/usr/bin/env python3
"""Append a complete recorded-dependency cost ledger to saved predictions."""
from pathlib import Path
import argparse
import csv
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4.costs import recorded_source_cost
from fk4.io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--prior-benchmark',type=Path);args=p.parse_args()
    prior=None
    if args.prior_benchmark:
        prior=json.loads((args.prior_benchmark/'summary.json').read_text())
        assert prior['status']=='complete' and all(r['pool_bitwise_equal'] for r in prior['rows'])
    config=json.loads((args.run/'config.json').read_text());s=json.loads((args.run/'summary.json').read_text())
    assert s['status']=='complete'
    frozen=Path(config['frozen']);f=json.loads((frozen/'summary.json').read_text())
    frozen_config=json.loads((frozen/'config.json').read_text())
    entries={r['key']:r for r in f['rows']};cache={};rows=[]
    for r in s['rows']:
        source=r['source_generation_seconds_cold'];ledger_key=None
        if r['method'].startswith('replica_'):
            entry=entries[r['frozen_source_key']];ledger_key=entry['source_run']
            if ledger_key not in cache:cache[ledger_key]=recorded_source_cost(ledger_key)
            source=cache[ledger_key]['recorded_seconds']
        elif r['method']=='physics_prior' and prior is not None:
            entry=entries[r['frozen_source_key']]
            if entry.get('source_cost_scope','').startswith('Independent analytic prior'):
                ledger_key='measured_independent_analytic_prior'
            else:
                source=prior['mean_generation_seconds']
                ledger_key='remeasured_identical_analytic_prior'
        elif r['method']=='dual_columns':
            ledger_key=str(Path(frozen_config['columns']).resolve())
            if ledger_key not in cache:cache[ledger_key]=recorded_source_cost(ledger_key)
            source=cache[ledger_key]['recorded_seconds']
        total=source+r['dictionary_generation_seconds']+r['dictionary_selection_seconds']+r['prediction_seconds']
        rows.append({'key':r['key'],'t':r['t'],'epsilon':r['epsilon'],'method':r['method'],'actual_gaussian_terms':r['actual_gaussian_terms'],
                     'original_recorded_cold_total':r['total_cold_prediction_seconds'],
                     'audited_recorded_source_seconds':source,'candidate_generation_seconds':r['dictionary_generation_seconds'],
                     'selection_seconds':r['dictionary_selection_seconds'],'prediction_seconds':r['prediction_seconds'],
                     'audited_recorded_cold_total':total,'ledger_key':ledger_key})
    shared={}
    for ledger in cache.values():
        for entry in ledger['unique_source_records']:
            if entry['directory'] in shared:assert shared[entry['directory']]==entry
            shared[entry['directory']]=entry
    shared_seconds=sum(entry['recorded_seconds'] for entry in shared.values())
    recorded_plan=shared_seconds+f['elapsed_seconds']+f.get('analytic_prior_setup_seconds',0.)+s['elapsed_seconds']
    write_json(args.run/'cost_audit.json',{'source_ledgers':cache,'rows':rows,
        'shared_source_dependency_union':list(shared.values()),
        'all_methods_shared_source_recorded_seconds':shared_seconds,
        'all_methods_recorded_source_freeze_prediction_seconds':recorded_plan,
        'whole_comparison_timing_scope':'Each unique upstream run once, plus measured freeze and prediction script walltimes and explicitly timed prior setup. Not a sum of per-prediction cold costs; concurrent source walltimes are summed. Independent verification/development and unrecorded setup are excluded.',
        'prediction_script_measured_wall_seconds':s['elapsed_seconds'],
        'frozen_preparation_measured_wall_seconds':f['elapsed_seconds'],
        'original_measurements_preserved':True,
        'prior_remeasurement':prior,
        'scope':'Recorded dependency-complete cold compute costs; complete reused sampling/validation runs and exact moment construction counted once within each cold-start graph. Parallel source runtimes are summed, not interpreted as elapsed wall time.',
        'limits':'Interpreter startup/unrecorded setup excluded. Shared costs are not amortized. If supplied, the identical analytic prior uses a separately labelled later timing including initialization rather than the unrelated column-optimization run; original timing columns remain. This is not an equal-time/hardware benchmark.'})
    with (args.run/'cost_audited_rows.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups={}
    for r in rows:groups.setdefault((r['t'],r['epsilon'],r['method'],r['actual_gaussian_terms']),[]).append(r)
    aggregated=[]
    for (t,eps,method,count),group in sorted(groups.items()):
        aggregated.append({'t':t,'epsilon':eps,'method':method,'actual_gaussian_terms':count,
            **{name:float(np.mean([r[name] for r in group])) for name in ['audited_recorded_source_seconds','candidate_generation_seconds','selection_seconds','prediction_seconds','audited_recorded_cold_total']}})
    with (args.run/'cost_audited_aggregates.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(aggregated[0]));writer.writeheader();writer.writerows(aggregated)
    report=args.run/'REPORT.md'
    addition='\n## 依赖计时审计\n\n请使用 `cost_audited_aggregates.csv` 的成本列。审计递归计入前阶witness、冻结验证来源及精确矩构造，并在一次冷启动依赖图中去重；原始预测数组和旧计时列保留。`cost_audit.json` 列出每一项来源。总和是记录到的计算时间，包含共享上游工作；不等于多卡并行墙钟时间，不包含未计时的进程启动/初始化，也不声称是等时预算比较。\n'
    if report.exists() and '## 依赖计时审计' not in report.read_text():report.write_text(report.read_text()+addition)
    print({key:ledger['recorded_seconds'] for key,ledger in cache.items()})


if __name__=='__main__':main()
