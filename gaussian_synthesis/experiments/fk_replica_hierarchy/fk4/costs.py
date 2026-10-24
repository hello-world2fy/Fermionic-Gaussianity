"""Trace recorded upstream construction/sampling costs without double counts."""
from pathlib import Path
import json


def recorded_source_cost(directory):
    seen=set();records=[];missing=[]
    def visit(directory):
        d=Path(directory).resolve()
        if str(d) in seen:return
        seen.add(str(d))
        config=json.loads((d/'config.json').read_text()) if (d/'config.json').exists() else {}
        if (d/'summary.json').exists():
            s=json.loads((d/'summary.json').read_text())
            if s.get('assembled_index_not_new_measurements'):
                for entry in s['provenance']:visit(entry['directory'])
                return
            if 'source_run' in s and 'reanalysis_seconds' in s:
                visit(s['source_run']);seconds=s['reanalysis_seconds'];kind='reanalysis'
            else:
                seconds=s.get('elapsed_seconds');kind='complete_upstream_run'
            if seconds is None:missing.append(str(d))
            else:records.append({'directory':str(d),'kind':kind,'recorded_seconds':float(seconds)})
        elif (d/'proof.json').exists():
            proof=json.loads((d/'proof.json').read_text())
            if 'elapsed_seconds' in proof:records.append({'directory':str(d),'kind':'exact_moment_construction','recorded_seconds':float(proof['elapsed_seconds'])})
            else:missing.append(str(d))
        else:missing.append(str(d))
        for name in ['initial_from','frozen','canonical','exact']:
            if config.get(name):visit(config[name])
        for name in ['moments','cartan_anchors']:
            for dependency in config.get(name,[]):visit(dependency)
    visit(directory)
    return {'recorded_seconds':sum(r['recorded_seconds'] for r in records),
            'unique_source_records':records,'missing_elapsed_records':missing,
            'scope':'Nonamortized sum of recorded full upstream runs plus reused exact-moment construction, with each directory counted once per cold start.',
            'unmeasured_overhead':'Interpreter startup and any setup outside each source timer are not reconstructed; this is a recorded-compute ledger, not a matched hardware/time-budget race.'}
