"""Append-only run artifacts with both new and reused source fingerprints."""
import hashlib
from pathlib import Path
from .paths import ROOT, PREVIOUS
from fk4.io import write_json, source_manifest as predecessor_manifest


def source_manifest(snapshot=None):
    result=predecessor_manifest(snapshot)
    for p in ROOT.rglob('*.py'):
        if 'results' in p.parts:continue
        name=str(p.relative_to(ROOT.parent));data=p.read_bytes()
        result['source_sha256'][name]=hashlib.sha256(data).hexdigest()
        if snapshot is not None:
            out=Path(snapshot)/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    result['current_spec_sha256']=hashlib.sha256((ROOT/'EXPERIMENT_SPEC.txt').read_bytes()).hexdigest()
    result['read_only_predecessor']=str(PREVIOUS)
    return result
